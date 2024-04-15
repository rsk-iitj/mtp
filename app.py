import streamlit as st
from nlp_processing.file_reader import extract_text_from_file
from nlp_tasks.summarization import summarize_text
from nlp_tasks.keyword_extraction import extract_keywords
from nlp_tasks.sentiment_analysis import analyze_sentiment
from nlp_tasks.generate_test_plan import generate_test_plan

st.title('Automated Test Plan Generator')

# Domain selection
domains = [
    "Telecom Industry", "E-Commerce", "IT Industry", "Marketing, Advertising, Sales", "Government sector",
    "Media & Entertainment", "Travel & Tourism", "IoT & Geofencing", "Finances",
    "Supply Chain, Inventory & Order Management", "Health Care, Fitness & Recreation",
    "Social Media, Social Media Analysis", "Ticketing", "Service Sector", "Gaming Industry",
    "Education Industry", "Mobile App Development", "Distribution Management System",
    "Science & Innovation", "Construction & Engineering", "Manufacturing",
    "Ecology and Environmental Protection", "Project Management Industry", "Logistics",
    "Procurement Management Solution", "Digital Agriculture"
]
selected_domain = st.selectbox("Select the application domain:", domains)

# Detailed tech stack inputs
tech_stack_options = {
    "Frontend Technology": ["React", "Angular", "Vue", "Other"],
    "Backend Technology": ["Node.js", "Python", "Java", "Other"],
    "Database": ["MySQL", "MongoDB", "PostgreSQL", "Other"],
    "Messaging Queue": ["RabbitMQ", "Kafka", "AWS SQS", "Other"],
    "Cloud Infrastructure": ["AWS", "Azure", "Google Cloud", "Other"]
}
tech_stack = {key: st.selectbox(f"Select {key}", options) for key, options in tech_stack_options.items()}
additional_tech = st.text_input("Specify any additional technologies not listed above:")

# Automation and resources
test_automation_required = st.radio("Is Test Automation Required?", ('Yes', 'No'))
num_testers = st.number_input("Number of Testers", min_value=0, value=0, step=1)
num_automation_testers = st.number_input("Number of Automation Testers", min_value=0, value=0, step=1)
num_test_lead = st.number_input("Number of Test Leads", min_value=0, value=0, step=1)
num_test_managers = st.number_input("Number of Test Managers", min_value=0, value=0, step=1)

# File uploader and user stories extraction
uploaded_files = st.file_uploader("Upload user stories documents", accept_multiple_files=True, type=['pdf', 'docx', 'doc', 'txt'])
if uploaded_files:
    combined_text = []
    for uploaded_file in uploaded_files:
        file_text = extract_text_from_file(uploaded_file)
        combined_text.append(file_text)
    user_stories_text = "\n\n".join(combined_text)
    st.text_area("Extracted User Stories", user_stories_text, height=300)

    # NLP processing
    summary = summarize_text(user_stories_text)
    keywords = extract_keywords(user_stories_text)
    sentiment_label, sentiment_score = analyze_sentiment(user_stories_text)

    if st.button("Generate Test Plan"):
        options = {
            'domain': selected_domain, 'tech_stack': tech_stack, 'additional_tech': additional_tech,
            'test_automation': test_automation_required, 'num_testers': num_testers,
            'num_automation_testers': num_automation_testers, 'num_test_lead': num_test_lead,
            'num_test_managers': num_test_managers,
            'keywords': ', '.join(keywords), 'sentiment': sentiment_label
        }

        test_plan = generate_test_plan(summary, keywords, sentiment_label)
        st.subheader("Generated Test Plan")
        st.write(test_plan)

else:
    st.write("Please upload at least one document containing user stories.")
