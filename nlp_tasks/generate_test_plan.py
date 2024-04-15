from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6B')

# Set padding token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_section_text(model, tokenizer, section, summary, keywords, sentiment_label):
    """Generate text for a given section of the test plan using a GPT model."""
    prompt = f"Create a concise, informative section titled '{section}' for a test plan. Focus on: {summary}, using keywords: {', '.join(keywords)}. The overall sentiment is {sentiment_label}."
    inputs = tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=tokenizer.model_max_length - 50)

    # Generate output ensuring not to exceed total max length
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=tokenizer.model_max_length,  # Ensures the total output length includes the input length
        max_new_tokens=50,  # Limits the additional generated tokens
        pad_token_id=tokenizer.pad_token_id
    )
    section_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return section_text

def generate_test_plan(summary, keywords, sentiment_label):
    sections = [
        "Test Plan Identifier", "References", "Introduction", "Test Items", "Software Risk Issues",
        "Features to be Tested", "Features not to be Tested", "Approach", "Item Pass/Fail Criteria",
        "Suspension Criteria and Resumption Requirements", "Test Deliverables", "Remaining Test Tasks",
        "Test Data Needs", "Environmental Needs", "Staffing and Training Needs", "Responsibilities",
        "Schedule", "Planning Risks and Contingencies", "Approvals", "Glossary", "Test Estimation"
    ]

    # Begin document
    test_plan_document = "Generated Test Plan\n\n"

    # Iterate over sections and generate content
    for section in sections:
        section_content = generate_section_text(model, tokenizer, section, summary, keywords, sentiment_label)
        test_plan_document += f"{section}:\n{section_content}\n\n"

    return test_plan_document
