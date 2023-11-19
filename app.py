from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the saved model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("C:/Users/saiha/OneDrive/Desktop/save files - text summarization/path_to_save_model")
model = T5ForConditionalGeneration.from_pretrained("C:/Users/saiha/OneDrive/Desktop/save files - text summarization/path_to_save_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/', methods=['GET', 'POST'])
def summarize():
    generated_summary = None

    if request.method == 'POST':
        user_input = request.form['user_input']  # Get user input from the HTML form

        # Tokenize the user input
        input_ids = tokenizer.encode(user_input, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(device)

        # Generate predictions using the model
        with torch.no_grad():
            output = model.generate(input_ids, max_length=128, num_return_sequences=1, early_stopping=True)

        # Decode the generated output text
        generated_summary = tokenizer.decode(output[0], skip_special_tokens=True)

    return render_template('index.html', generated_summary=generated_summary)

if __name__ == '__main__':
    app.run(debug=True)
