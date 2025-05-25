import json
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

def analyze_human_annotation_trends(json_filepath, csv_filepath):
    """
    Analyzes the trends of human annotation guidelines over the years,
    separated by rubric, based on the provided JSON and CSV file paths.
    It dynamically collects all categories within 'response' that contain an 'is_applicable' field.

    Args:
        json_filepath (str): The file path to the JSON file.
        csv_filepath (str): The file path to the CSV file.

    Returns:
        dict: A nested dictionary containing yearly counts for each discovered category,
              grouped by rubric ID.
              The structure is {rubric_id: {year: {'DiscoveredCategoryName': count, ...}}}.
    """
    # Load the CSV content using pandas
    try:
        csv_df = pd.read_csv(csv_filepath)
        # Create a mapping from id to year from the DataFrame
        id_to_year = csv_df.set_index('id')['year'].astype(str).to_dict()
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}")
        return {}
    except KeyError:
        print("Error: 'id' or 'year' column not found in CSV file.")
        return {}
    except Exception as e:
        print(f"Error loading or processing CSV file: {e}")
        return {}

    # Load the JSON content using json.load
    try:
        with open(json_filepath, 'r') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_filepath}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_filepath}. Check file format.")
        return {}
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return {}

    # Initialize a nested dictionary to store yearly counts per category, per rubric
    # Structure: {rubric_id: {year: {'DiscoveredCategoryName': count, ...}}}
    rubric_yearly_guideline_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Iterate through the JSON data entries
    for entry in json_data:
        full_json_id = entry.get('id')
        if full_json_id:
            # Find the index of the '-rubric-' substring
            rubric_separator_index = full_json_id.find('-rubric-')

            if rubric_separator_index != -1:
                # Extract the base ID as everything before '-rubric-'
                base_json_id = full_json_id[:rubric_separator_index]
                # Extract the rubric ID as everything from '-rubric-' onwards
                rubric_id = full_json_id[rubric_separator_index + 1:]
            else:
                # If '-rubric-' is not found, the entire ID is the base ID,
                # and we assign a default rubric ID.
                base_json_id = full_json_id
                rubric_id = 'no-rubric-suffix' # Default if no '-rubric-' is found

            year = id_to_year.get(base_json_id)
            if year:
                # Convert year to an integer for proper numerical sorting and plotting
                try:
                    year = int(year)
                except ValueError:
                    # Skip this entry if the year value is not a valid integer
                    continue

                response_data = entry.get('response', {})

                # Dynamically iterate through all top-level keys within 'response_data'
                # (e.g., 'human_annot_guidelines', 'non_human_language_content', etc.)
                for section_name, section_content in response_data.items():
                    # Check if the value associated with the top-level key is a dictionary
                    # and contains nested dictionaries with an 'is_applicable' field.
                    if isinstance(section_content, dict):
                        # Iterate through the items of this section (which are the categories like 'N/A', 'Has Instructions')
                        for category_name, category_details in section_content.items():
                            # Check if 'category_details' is a dictionary and contains 'is_applicable'
                            if isinstance(category_details, dict) and 'is_applicable' in category_details:
                                # If 'is_applicable' is True, then count this category for the current rubric and year
                                if category_details.get('is_applicable') is True:
                                    # Store the count using the 'category_name' (e.g., 'N/A', 'Has Instructions', etc.)
                                    # The plot_guideline_trends function will then dynamically get these categories.
                                    rubric_yearly_guideline_counts[rubric_id][year][category_name] += 1
                                else:
                                    rubric_yearly_guideline_counts[rubric_id][year][category_name] += 0

    return rubric_yearly_guideline_counts

def plot_guideline_trends(rubric_data):
    """
    Generates individual bar plots for each rubric ID, displaying the trends
    of discovered categories over the years. The categories plotted
    are dynamically determined for each rubric based on the collected data.
    Each plot is saved to a separate file.

    Args:
        rubric_data (dict): A nested dictionary containing yearly counts for
                            each discovered category, grouped by rubric ID,
                            as returned by analyze_human_annotation_trends.
    """
    if not rubric_data:
        print("No data available to plot for human annotation guidelines across any rubric.")
        return

    # Iterate through each rubric and generate a separate plot
    for rubric_id, yearly_data_for_rubric in rubric_data.items():
        if not yearly_data_for_rubric:
            print(f"No data to plot for rubric: {rubric_id}")
            continue

        # Collect all unique category names that appeared for this specific rubric
        # across all years. This will ensure all categories are present in the DataFrame
        # columns and thus in the legend.
        all_unique_categories_for_rubric = set()
        for year_data in yearly_data_for_rubric.values():
            all_unique_categories_for_rubric.update(year_data.keys())
        all_unique_categories_for_rubric = sorted(list(all_unique_categories_for_rubric))


        # Create DataFrame directly with years as index and categories as columns
        df = pd.DataFrame.from_dict(yearly_data_for_rubric, orient='index')

        # Reindex the DataFrame to include all unique categories as columns.
        # This ensures that even categories with zero counts are present,
        # allowing them to appear in the legend. Fill missing values with 0.
        df = df.reindex(columns=all_unique_categories_for_rubric, fill_value=0)

        # Ensure all values are integers
        df = df.astype(int)
        # Sort the DataFrame by year (index) to ensure chronological order on the plot
        df = df.sort_index()

        # Get the actual categories (column names) for this specific rubric's plot.
        current_rubric_categories = df.columns.tolist()

        # Dynamically adjust figure width based on number of years and categories
        num_years = len(df.index)
        num_categories = len(current_rubric_categories)
        
        # Base width per year group, plus extra for more categories
        base_width_per_year = 1.5
        extra_width_per_category = 0.3
        
        # Calculate dynamic figure width
        dynamic_fig_width = num_years * base_width_per_year + num_categories * extra_width_per_category
        # Set a minimum width to ensure readability for fewer data points
        dynamic_fig_width = max(dynamic_fig_width, 8) 

        fig, ax = plt.subplots(figsize=(dynamic_fig_width, 8)) # Use dynamic width, fixed height

        # Set the width of each bar
        bar_width = 0.8 / (num_categories + 1) # Adjust bar width based on number of categories
        if bar_width > 0.3: # Cap max bar width to avoid them being too thick
            bar_width = 0.3

        # Create an array of x-coordinates for the groups of bars
        index = range(num_years) # Use range for indexing

        # Plot grouped bars for each category present in this rubric's data
        for i, category in enumerate(current_rubric_categories):
            counts = df[category]
            # Calculate the position for each bar within its group
            # Adjust bar positions to be centered within their year group
            bar_positions = [x + i * bar_width - (num_categories - 1) * bar_width / 2 for x in index]
            bars = ax.bar(bar_positions, counts, bar_width, label=category)

            # Add the count number on top of each bar
            for bar in bars:
                yval = bar.get_height()
                if yval > 0: # Only display count if it's greater than 0
                    # Adjust text position slightly above the bar
                    ax.text(bar.get_x() + bar.get_width()/2, yval + (ax.get_ylim()[1] * 0.01), # Dynamic offset
                            int(yval), ha='center', va='bottom', fontsize=9, color='black')


        # Set plot labels and title for the current rubric's plot
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Papers', fontsize=12)
        # Dynamically set title based on the rubric ID
        ax.set_title(f'Trends in {rubric_id} Guidelines Over Years (NeurIPS)', fontsize=14)

        # Set x-axis ticks and labels
        if current_rubric_categories:
            # Set x-ticks to be at the center of each year's group of bars
            ax.set_xticks(index)
            ax.set_xticklabels(df.index, rotation=45, ha="right") # Rotate labels for readability
        else:
            ax.set_xticks([]) # No ticks if no data

        # Add a legend to identify each bar type.
        ax.legend(title="Category Type")

        # Add a grid for better readability of values
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Adjust layout to prevent labels from overlapping and provide more space
        plt.tight_layout(pad=3.0) # Added padding around the plot

        # Save the plot to a file instead of just displaying it
        # The filename will include the rubric ID
        plot_filename = f'trends_{rubric_id}.png'
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}") # Inform the user about the saved file

        # Close the plot to free up memory, especially important when generating many plots
        plt.close(fig)

# To use these functions, you would now provide the file paths:
#
# json_file_path = 'neurips_qwen3_32b_full.json'
# csv_file_path = 'filtered_year_neurips_conference.csv'
#
# # 1. Analyze the data
# trends_data_by_rubric = analyze_human_annotation_trends(json_file_path, csv_file_path)
#
# # 2. Plot the trends for each rubric (this will now save files)
# plot_guideline_trends(trends_data_by_rubric)


json_file = r'C:\Users\Default.DESKTOP-4PK7F2T\Desktop\datarubrics-dev\data\outputs\nlp_phi4.json'
csv_file = r'C:\Users\Default.DESKTOP-4PK7F2T\Desktop\datarubrics-dev\data\csv\filtered_year_nlp_conference.csv'

# json_file = r'C:\Users\Default.DESKTOP-4PK7F2T\Desktop\datarubrics-dev\data\outputs\cv_out_qwen3_32b.json'
# csv_file = r'C:\Users\Default.DESKTOP-4PK7F2T\Desktop\datarubrics-dev\data\csv\filtered_year_cv_conference.csv'

trends_data = analyze_human_annotation_trends(json_file, csv_file)

plot_guideline_trends(trends_data)