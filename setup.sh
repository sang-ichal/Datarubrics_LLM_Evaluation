uv venv
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    .venv\Scripts\activate
else
    source .venv/bin/activate
fi
uv pip sync requirements.txt