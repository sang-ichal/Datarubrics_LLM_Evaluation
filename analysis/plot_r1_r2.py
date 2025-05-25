import json
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker # Import for percentage formatting

def analyze_specific_rubric_trends(json_filepaths, csv_filepaths, conference_names, target_rubrics):
    """
    Analyzes trends for specific rubrics across multiple conferences,
    aggregating all categories except 'N/A' into an 'All Other Categories (OR)' group.

    Args:
        json_filepaths (list): A list of file paths to the JSON files, one for each conference.
        csv_filepaths (list): A list of file paths to the CSV files, one for each conference.
        conference_names (list): A list of names for each conference, corresponding to file paths.
        target_rubrics (list): A list of rubric IDs (e.g., ['rubric-1', 'rubric-2'])
                                for which to apply the specific aggregation logic.

    Returns:
        dict: A nested dictionary containing yearly counts for 'N/A' and
              'All Other Categories (OR)', grouped by conference name and then by target rubric ID.
              Structure: {conference_name: {rubric_id: {year: {'N/A': count, 'All Other Categories (OR)': count}}}}.
    """
    if not (len(json_filepaths) == len(csv_filepaths) == len(conference_names)):
        print("Error: The lists of JSON file paths, CSV file paths, and conference names must be of the same length.")
        return {}

    all_conferences_rubric_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))

    for conf_idx, conf_name in enumerate(conference_names):
        json_filepath = json_filepaths[conf_idx]
        csv_filepath = csv_filepaths[conf_idx]

        print(f"Processing data for conference: {conf_name}")

        # Load the CSV content using pandas
        try:
            csv_df = pd.read_csv(csv_filepath)
            id_to_year = csv_df.set_index('id')['year'].astype(str).to_dict()
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_filepath} for {conf_name}. Skipping this conference.")
            continue
        except KeyError:
            print(f"Error: 'id' or 'year' column not found in CSV file {csv_filepath} for {conf_name}. Skipping this conference.")
            continue
        except Exception as e:
            print(f"Error loading or processing CSV file {csv_filepath} for {conf_name}: {e}. Skipping this conference.")
            continue

        # Load the JSON content using json.load
        try:
            with open(json_filepath, 'r') as f:
                json_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {json_filepath} for {conf_name}. Skipping this conference.")
            continue
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {json_filepath} for {conf_name}. Check file format. Skipping this conference.")
            continue
        except Exception as e:
            print(f"Error loading JSON file {json_filepath} for {conf_name}: {e}. Skipping this conference.")
            continue

        # Process data for the current conference
        for entry in json_data:
            full_json_id = entry.get('id')
            if full_json_id:
                rubric_separator_index = full_json_id.find('-rubric-')

                if rubric_separator_index != -1:
                    base_json_id = full_json_id[:rubric_separator_index]
                    rubric_id = full_json_id[rubric_separator_index + 1:]
                else:
                    base_json_id = full_json_id
                    rubric_id = 'no-rubric-suffix'

                # Only process if the rubric_id is one of the target rubrics
                if rubric_id in target_rubrics:
                    year = id_to_year.get(base_json_id)
                    if year:
                        try:
                            year = int(year)
                        except ValueError:
                            continue

                        response_data = entry.get('response', {})
                        if not response_data:
                            continue

                        any_other_category_applicable = False

                        for section_name, section_content in response_data.items():
                            if isinstance(section_content, dict):
                                for category_name, category_details in section_content.items():
                                    if isinstance(category_details, dict) and 'is_applicable' in category_details:
                                        if category_details.get('is_applicable') is True:
                                            if category_name == 'N/A':
                                                all_conferences_rubric_data[conf_name][rubric_id][year]['N/A'] += 1
                                            else:
                                                any_other_category_applicable = True
                        
                        if any_other_category_applicable:
                            all_conferences_rubric_data[conf_name][rubric_id][year]['All Other Categories (OR)'] += 1
    
    return all_conferences_rubric_data

def plot_specific_rubric_trends(all_conferences_rubric_data, target_rubrics):
    """
    Generates individual line plots for specific rubric IDs, displaying trends
    for 'All Other Categories (OR)' across multiple conferences.
    The 'N/A' category is used for percentage calculation but not plotted.
    Each plot is saved to a separate file.
    Numbers on top of line points are displayed as raw counts with percentages in parentheses.
    The Y-axis is displayed as percentages.

    Args:
        all_conferences_rubric_data (dict): A nested dictionary containing yearly counts for
                                            'N/A' and 'All Other Categories (OR)', grouped by
                                            conference name and then by rubric ID,
                                            as returned by analyze_specific_rubric_trends.
        target_rubrics (list): The list of rubric IDs that were analyzed.
    """
    if not all_conferences_rubric_data:
        print("No data available to plot for the specified rubrics across any conference.")
        return

    # Define the categories for data processing (both are needed for total_year_count)
    data_categories = ['All Other Categories (OR)', 'N/A']
    # Define the categories to actually plot (only 'All Other Categories (OR)')
    plot_categories = ['All Other Categories (OR)']

    # Define a color map for conferences for consistent plotting
    colors = plt.cm.get_cmap('tab10', len(all_conferences_rubric_data))
    conference_colors = {conf_name: colors(i) for i, conf_name in enumerate(all_conferences_rubric_data.keys())}

    # Iterate through each target rubric to create a plot
    for rubric_id in target_rubrics:
        rubric_has_data = False
        for conf_name in all_conferences_rubric_data:
            if rubric_id in all_conferences_rubric_data[conf_name] and all_conferences_rubric_data[conf_name][rubric_id]:
                rubric_has_data = True
                break
        
        if not rubric_has_data:
            print(f"No aggregated data found for rubric: {rubric_id} across any conference. Skipping plot.")
            continue

        all_years = set()
        for conf_name in all_conferences_rubric_data:
            if rubric_id in all_conferences_rubric_data[conf_name]:
                all_years.update(all_conferences_rubric_data[conf_name][rubric_id].keys())
        all_years = sorted(list(all_years))
        num_years = len(all_years)

        if num_years == 0:
            print(f"No year data found for rubric: {rubric_id}. Skipping plot.")
            continue

        # Create the figure with a single subplot for 'All Other Categories (OR)'
        fig, ax1 = plt.subplots(1, 1, figsize=(11, 6.5)) # Increased width for legend

        # Removed fig.suptitle()
        ax1.set_title(f'Trends in {rubric_id} Guidelines Over Years (Aggregated by Conference)') # Moved title to axes

        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Percentage of Papers', fontsize=12) # Changed Y-axis label
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Set Y-axis to display percentages
        ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

        # Iterate through each conference to plot its data
        for conf_idx, conf_name in enumerate(all_conferences_rubric_data.keys()):
            conf_color = conference_colors[conf_name]
            
            # Prepare data for the current conference and rubric
            conf_df_data = {year: all_conferences_rubric_data[conf_name][rubric_id].get(year, defaultdict(int)) for year in all_years}
            conf_df = pd.DataFrame.from_dict(conf_df_data, orient='index')
            # Ensure both data_categories are present for total sum calculation
            conf_df = conf_df.reindex(columns=data_categories, fill_value=0).fillna(0).astype(int)
            conf_df = conf_df.sort_index()

            # Plot 'All Other Categories (OR)' as a line
            if 'All Other Categories (OR)' in conf_df.columns:
                # Get the relevant data for plotting
                years_to_plot = conf_df.index.tolist()
                raw_counts_to_plot = conf_df['All Other Categories (OR)'].tolist()
                
                # Calculate percentages for plotting and labeling
                percentages_to_plot = []
                total_counts_for_years = conf_df[data_categories].sum(axis=1).tolist()
                for i, count in enumerate(raw_counts_to_plot):
                    total = total_counts_for_years[i]
                    percentage = count / total if total > 0 else 0
                    percentages_to_plot.append(percentage)

                # Plot the line
                ax1.plot(years_to_plot, percentages_to_plot, marker='o', linestyle='-', color=conf_color, label=conf_name)
                
                # Add raw count and percentage text on top of each point
                for i, year_value in enumerate(years_to_plot):
                    raw_count = raw_counts_to_plot[i]
                    percentage = percentages_to_plot[i]
                    
                    if raw_count > 0:
                        ax1.text(year_value, percentage + (ax1.get_ylim()[1] * 0.02), # Dynamic offset
                                f'{int(raw_count)}', ha='center', va='bottom', fontsize=12, color='black')

        # Set x-ticks and labels for the subplot
        ax1.set_xticks(all_years)
        ax1.set_xticklabels(all_years, rotation=45, ha="right")

        # Create the legend directly on the axes (ax1)
        ax1.legend(title="Conference", loc='upper left', bbox_to_anchor=(1.02, 1))

        # Use tight_layout with rect to make space for the legend on the right.
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjusted right boundary for legend space

        plot_filename = f'trends_{rubric_id}_multi_conference_aggregated.png'
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
        plt.close(fig)

# Example Usage:
# json_file_path = 'neurips_qwen3_32b_full.json'
# csv_file_path = 'filtered_year_neurips_conference.csv'
# target_rubrics_to_plot = ['rubric-1', 'rubric-2'] # Specify the rubrics you want to plot with this aggregation

# 1. Analyze the data with specific aggregation
# aggregated_trends = analyze_specific_rubric_trends(json_file_path, csv_file_path, target_rubrics_to_plot)

# 2. Plot the trends for the specified rubrics
# plot_specific_rubric_trends(aggregated_trends)

target_rubrics_to_plot = ['rubric-1', 'rubric-2']
json_file_paths = [r'C:\Users\Default.DESKTOP-4PK7F2T\Desktop\datarubrics-dev\data\outputs\nlp_phi4.json', 
                   r'C:\Users\Default.DESKTOP-4PK7F2T\Desktop\datarubrics-dev\data\outputs\cv_out_qwen3_32b.json']
csv_file_paths = [r'C:\Users\Default.DESKTOP-4PK7F2T\Desktop\datarubrics-dev\data\csv\filtered_year_nlp_conference.csv', 
                  r'C:\Users\Default.DESKTOP-4PK7F2T\Desktop\datarubrics-dev\data\csv\filtered_year_cv_conference.csv']

conference_names = ['NLP', 'CV']

aggregated_trends = analyze_specific_rubric_trends(json_file_paths, csv_file_paths, conference_names, target_rubrics_to_plot)

plot_specific_rubric_trends(aggregated_trends, target_rubrics_to_plot)