import streamlit as st
import re
import random

from src.generator import MetricsGenerator, MetricsGenerationConfig, AppLabel


MODEL2NAME = {
    "gpt-4o": "OpenAI",
    "claude-3.7": "Anthropic"
}

# Sample response to simulate LLM output
def generate_metrics(description, selected_labels, model_name, api_key):
    print(f"Using {MODEL2NAME[model_name]} API key: {api_key[:3]}...{api_key[-3:]} for model: {model_name}")

    # This would be replaced with your actual LLM call
    metrics = []
    
    config = MetricsGenerationConfig(
        app_description = description,
        app_labels = selected_labels,
        # model_name = "gpt-4o",
        model_name = model_name, #"gpt-4o",
        api_key = api_key,
    )
    generator = MetricsGenerator(config)
    generated_metrics = generator.generate()
    
    return generated_metrics.metrics

# Initialize session state variables
if 'metrics' not in st.session_state:
    st.session_state.metrics = []
    
if 'show_code_for' not in st.session_state:
    st.session_state.show_code_for = None
    
if 'has_generated' not in st.session_state:
    st.session_state.has_generated = False

if 'api_keys' not in st.session_state:
    # Dictionary to store API keys for different models
    st.session_state.api_keys = {
        "gpt-4o": None,
        "claude-3.7": None
    }
    
if 'show_api_modal' not in st.session_state:
    st.session_state.show_api_modal = False

# Function to handle code view button click
def view_code(index):
    st.session_state.show_code_for = index

# Function to handle submit button
def on_submit():
    # Check if we have an API key for the selected model
    selected_model = st.session_state.selected_model
    
    if st.session_state.api_keys[selected_model] is None:
        # We need to get the API key for this model
        st.session_state.show_api_modal = True
    else:
        # We already have the API key, proceed with generation
        generate_metrics_with_key()

# Function to save API key and proceed
def save_api_key():
    api_key = st.session_state.api_key_input
    selected_model = st.session_state.selected_model
    
    if api_key and api_key.strip():
        # Save the API key for this specific model
        st.session_state.api_keys[selected_model] = api_key
        st.session_state.show_api_modal = False
        # Now generate metrics
        generate_metrics_with_key()
    else:
        st.error("Please enter a valid API key")

# Function to actually generate metrics once we have an API key
def generate_metrics_with_key():
    st.session_state.has_generated = True

    selected_model = st.session_state.selected_model
    api_key = st.session_state.api_keys[selected_model]

    with st.spinner("Generating metrics..."):
        metrics = generate_metrics(
            st.session_state.app_description,
            st.session_state.selected_labels,
            selected_model,
            api_key
        )
    st.session_state.metrics = metrics

# Function to properly format code with escaped newlines
def format_code(code_str):
    # Replace escaped newlines with actual newlines
    formatted_code = code_str.replace('\\n', '\n')
    
    # Handle any additional escape characters
    formatted_code = re.sub(r'\\([^n])', r'\1', formatted_code)
    
    return formatted_code

# App title and description
st.title("Automatic Metric Generation")
# st.markdown("Demo for Galileo Auto-Metrics Generation.")

# Input section
st.header("Input Parameters")

# 0. API Key
# api_key = st.text_input("This demo makes calls to OpenAI. Enter the API key below (search for \"OpenAI API Key\" in 1Pass):", type="password")

# 1. Freeform text description
st.session_state.app_description = st.text_area(
    "Enter a brief description of your application:", 
    value="Ai assistant for a medical clinic. Does administrative tasks like scheduling apointments and followups, and registering new clients.",
    key="app_description_input"
)

# 2. Pregenerated labels
label_options = ["RAG", "Agent", "Code Generation", "Multi-Turn"]
st.session_state.selected_labels = st.multiselect(
    "Select relevant labels:", 
    label_options, 
    default=["Agent", "Multi-Turn"],
    key="labels_input"
)

# 3. Model selection
model_options = ["gpt-4o", "claude-3.7"]
st.session_state.selected_model = st.selectbox(
    "Select the model to use:", 
    model_options,
    key="model_input"
)

# Submit button
st.button("Generate Metrics", on_click=on_submit)

# API Key Modal
if st.session_state.show_api_modal:
    # Create a modal dialog using st.form
    with st.form(key="api_key_form"):
        st.subheader("Enter your API Key")
        st.text_input(
            f"API Key for {MODEL2NAME[st.session_state.selected_model]} (search for {MODEL2NAME[st.session_state.selected_model]} API Key in OnePass)",
            type="password",
            key="api_key_input"
        )
        st.markdown("Your API key is required to use the LLM service and will be stored only for this session.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.form_submit_button("Cancel", on_click=lambda: setattr(st.session_state, 'show_api_modal', False))
        with col2:
            st.form_submit_button("Submit", on_click=save_api_key)

# Results section
if st.session_state.has_generated:
    st.header("Generated Metrics")
    st.success(f"Generated {len(st.session_state.metrics)} metrics for your application!")
    
    # Display metrics in expandable cards
    for i, metric in enumerate(st.session_state.metrics):
        with st.expander(f"{metric.name} - {metric.implementation}"):
            st.write(f"**Description:** {metric.description}")
            st.write(f"**Implementation mode:** {metric.implementation}")
            
            # Code button only if implementation exists
            if metric.code:
                st.button(
                    "View Code Implementation", 
                    key=f"code_button_{i}",
                    on_click=view_code,
                    args=(i,)
                )



# Sidebar for code display
st.sidebar.title("Code Implementation")
if st.session_state.show_code_for is not None and st.session_state.metrics:
    index = st.session_state.show_code_for
    if 0 <= index < len(st.session_state.metrics):
        metric = st.session_state.metrics[index]
        
        if metric.code:
            st.sidebar.subheader(f"Code for: {metric.name}")
            st.sidebar.code(format_code(metric.code), language="python")
            
            # Option to hide code
            if st.sidebar.button("Hide Code"):
                st.session_state.show_code_for = None
        else:
            st.sidebar.info("No code implementation available for this metric.")
else:
    st.sidebar.info("Click 'View Code Implementation' on a metric to see its code here.")

# Option to reset API keys
st.subheader("API Key Management")
for model in st.session_state.api_keys:
    if st.session_state.api_keys[model] is not None:
        if st.button(f"Reset {MODEL2NAME[model]} API Key"):
            st.session_state.api_keys[model] = None
            st.success(f"{model} API key has been reset")