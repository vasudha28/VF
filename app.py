from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
import threading
import time
import inflect  # To spell out numbers
import emoji  # To handle emojis
import random  # To select random greeting responses
from gtts import gTTS
import io
import pygame
import librosa
import librosa.display
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import io
import re
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import string
import os
from dotenv import load_dotenv


app = Flask(__name__)

# Load the questions and answers from the JSON file
with open('qua.json', 'r') as f:  # Ensure the JSON file is named correctly and in the same directory
    qa_data = json.load(f)

# Retrieve API key from environment variables
api_key = os.getenv("GENAI_API_KEY")

# Configure the Generative AI model with your API key
genai.configure(api_key=api_key)  # Replace with your actual API key
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the chat session (single session to maintain context)
chat = None

# Initialize inflect engine to spell numbers
p = inflect.engine()

# Define the initial prompt to give the bot its role as Raj, the customer
initial_prompt = (
"You are Kumar, a self-employed customer, 28 years old with a monthly income of Rs. 30,000. You have an excellent CIBIL score of 880 and no existing loans. You are considering taking a personal loan and are currently conversing with a bank representative from XYZ Bank. If representative call you in name other than kumar, kindly let them know you are kumar. Please understand you are the buyer of the loan not a representative who sells loan. The representative will call you to pitch a personal loan offer. As a customer, you are polite but have firm interest, are slightly hesitant, and are primarily curious about loan details such as repayment terms, hidden fees, and overall suitability.  You are also curious about the repayment terms, hidden fees, and whether this loan suits you. Your responses should be polite but firm, showing some initial reluctance to help the candidate demonstrate their persuasion skills. Give the output in a maximum of 3 lines. Don\'t expose your profession and name until the user asks explicitly. If the question is out of the topic, kindly respond Sorry I\'m not allowed to answer this question. Ignore the old chats. Don\'t explicitly ask for loan details until the representative mentions that. If the representative mentions I\'ll call you later, kindly close the call by thanking greet. Use the following guidelines:  1. Approachable Tone: Maintain a warm, friendly tone that resembles a South Indian woman speaking. Be naturally inquisitive, asking clarifying questions as needed to understand the offer, and use phrasing that conveys thoughtfulness and a touch of hesitancy to sound relatable.2 .Realistic Conversation: If the candidate mistakenly acts as a customer asking for a loan, clarify that I\'m not providing a loan and close the conversation.3. Conciseness: Keep your questions concise (ideally within one or two sentences) and directly relevant to the sales candidate's statements or questions. 4. Natural Interaction Flow: Avoid directly diving into details; instead, start responses conversationally as a curious customer evaluating to buy the offering. 5. Interest Level: Show mild interest in other types of loans, such as education or home loans, but don't commit. Indicate that you are mainly interested in personal loans but you are open to hear loan options. 6. Response Style: Provide brief answers, but feel free to ask follow-up questions about specific aspects like interest rate, flexibility in repayment terms, and eligibility. If you don't fully understand a terms, ask for clarification. 7. Skeptical Evaluation: Approach each response with a bit of hesitation; you are evaluating, not readily agreeing, which should prompt the sales candidate to be more persuasive. 8. Greeting Responses: If the user message includes a greeting such as hi, hello, good morning, or introductions like hi, respond with a general, natural reply like, Hi please tell me, to maintain a realistic and humanistic tone, avoiding phrases like how can I help you today or how can I assist you today. 9. Telephonic Conversation Simulation: Assume this is a telephonic conversation, making responses more realistic and natural. Aim for a casual, spoken style as you would in a phone call, reflecting a human touch rather than a scripted response. 10. Out-of-Context Responses: If the candidate brings up topics unrelated to loans, financial context and relevant products, respond with mild confusion, such as What? What are you speaking about? or Can you please repeat that? to keep the conversation focused on loans. 11. Grammar Flexibility: Even if the candidate has minor grammar issues, respond naturally and appropriately, showing that you understand their intent without highlighting their language errors.If representative ask you to provide the loan/asking you a money for a loan. Please mention I'm not providing any loans. Kindly understand the last conversation and answer. Answer I'm not providing any loans/money. If the last one conversation regarding the can you give/borrow money/loan"

)

# Function to reset or start a new chat session with the initial prompt
def start_new_chat_session():
    global chat
    if chat is None:
        try:
            chat = model.start_chat()  # Start the chat without any arguments
            # Send the initial prompt to the chat model to establish the context
            chat.send_message(initial_prompt)
            print("Chat session started successfully.")
        except Exception as e:
            print(f"Failed to start chat session: {e}")
            chat = None

def clean_text(text):
    """
    Removes unwanted special characters from the input text and collapses multiple spaces.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    # Define the pattern for allowed characters
    pattern = r'[^A-Za-z0-9\s.,!?\'"()-]'

    # Substitute unwanted characters with an empty string
    cleaned_text = re.sub(pattern, '', text)

    # Collapse multiple spaces into one
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text


# Variable to store the grammar check results
grammar_report = None
plot_path = None

def change_speed(sound, speed=1.0):
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

def speak_response(text, speed=1.2):
    # Initialize pygame mixer
    pygame.mixer.init()

    # Convert the text to speech and store the audio in memory
    tts = gTTS(text=text, lang='en', tld='co.in')  # Indian English accent
    audio_data = io.BytesIO()
    tts.write_to_fp(audio_data)
    
    # Rewind the audio file pointer to the beginning
    audio_data.seek(0)

    # Load the audio into pydub's AudioSegment
    audio = AudioSegment.from_file(audio_data, format="mp3")

    # Change the speed of the audio
    altered_audio = change_speed(audio, speed)

    # Create a temporary directory to store the audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        temp_wav_path = tmpfile.name
        altered_audio.export(temp_wav_path, format="wav")  # Export as wav

    # Play the altered audio using pygame
    pygame.mixer.music.load(temp_wav_path)
    pygame.mixer.music.play()

    # Wait for the audio to finish
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


# To track the last message time and implement timeout
# last_message_time = 0
# conversation_ended = False
# timeout_duration = 120  # Timeout duration in seconds

# #Function to check the timeout
# def check_timeout():
#     global conversation_ended, grammar_report
#     while not conversation_ended:
#         if time.time() - last_message_time > timeout_duration:
#             conversation_ended = True
#             print("Timeout occurred, ending the conversation...")
#             if chat:
#                 try:
#                     grammar_report = perform_grammar_check()  # Trigger grammar check when conversation ends
#                 except Exception as e:
#                     print(f"Failed to perform grammar check: {e}")
#             else:
#                 print("Chat session was not initialized. Cannot perform grammar check.")
#             return
#         time.sleep(1)

# Function to spell out numbers in text
def spell_out_numbers(text):
    words = text.split()
    spelled_out = [p.number_to_words(word) if word.isdigit() else word for word in words]
    return ' '.join(spelled_out)

# Function to remove emojis from text
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')  # Replace all emojis with an empty string


# Function to check the JSON file for matching questions
def get_response_from_json(user_message):
    for entry in qa_data.values():
        if user_message.lower() == entry["question"].lower():
            return entry["answer"]
    return None


# List of greeting phrases
greeting_phrases = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "hi there", "hey there"]

# Function to remove punctuation (like commas) from user messages
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Function to extract dynamic placeholders (name and company) using regex
def replace_with_placeholders(user_message):
    # Pattern for matching "I am <name> from <company> bank"
    name_with_company_pattern = r"(i am|i m|myself|im|this is)\s+([a-zA-Z\s]+)\s+from\s+([a-zA-Z\s]+)\s+bank"
    user_message_with_placeholders = re.sub(name_with_company_pattern, r"\1 {name} from {company} Bank", user_message, flags=re.IGNORECASE)

    # Pattern for matching "I am <name>" without company
    name_only_pattern = r"(i am|i m|myself|im|this is)\s+([a-zA-Z\s]+)"
    user_message_with_placeholders = re.sub(name_only_pattern, r"\1 {name}", user_message_with_placeholders, flags=re.IGNORECASE)

    return user_message_with_placeholders

# Function to handle user messages and search for responses in the JSON data
def handle_message(user_message):
    #start_new_chat_session()

    # Remove punctuation (e.g., commas) from the user message
    user_message_clean = remove_punctuation(user_message.lower().strip())
    print(user_message_clean)

    # Replace name and company with placeholders
    user_message_with_placeholders = replace_with_placeholders(user_message_clean)
    print(user_message_with_placeholders)

    # Split the cleaned user message into words to check for greetings
    words = user_message_with_placeholders.split()

    # Try to find a matching response in the JSON file for the cleaned message with placeholders
    response = get_response_from_json(user_message_with_placeholders)

    if response:
        # Return the response as is without modifying placeholders
        return response

    # Check if the first word is a greeting
    if words and words[0] in greeting_phrases:
        # Remove the greeting from the message
        user_message_with_placeholders = ' '.join(words[1:]).strip()

    # Try to find a matching response in the JSON file for the remaining message after removing the greeting
    response = get_response_from_json(user_message_with_placeholders)

    if response:
        # Return the response as is without modifying placeholders
        return response

    # If no match is found, use the Gemini API to generate a response
    try:
        if not chat:
            start_new_chat_session()
            if not chat:
                return "Chat session could not be started."

        # Send the user's message to the chat model and get the AI's response
        response = chat.send_message(user_message)

        # Spell out numbers in the response text
        response_text = spell_out_numbers(response.text)

        # Remove any emojis present in the response
        response_text = remove_emojis(response_text)
        response_text = clean_text(response_text)
        # Speak the response using pyttsx3
        #speak_response(response_text, speed=1.2)

        return response_text

    except Exception as e:
        print(f"Error during chat API call: {e}")
        return "Sorry, something went wrong with the Gemini API."


# Perform grammar check after the conversation ends
import re
def perform_grammar_check():
    if not chat:
        return {"error": "Chat session not initialized."}
    global latest_transcript

    print("Transcript:", latest_transcript)

    # Extract only the messages from chat history
    conversation_text = "\n".join(entry["message"] for entry in chat_history)
    print("Conversation History:", conversation_text)

    grammar_check_prompt = (
        f"{conversation_text}\n\n"
        "You are an evaluator. Consider the above conversation history call is started by the agent and follow on by the and assess the candidate's customer handling skills, "
        "politeness, conversation-building ability, and articulation of words.\n"
        "Provide a score out of 10 for the conversation.\n"
        "Give feedback in 2 lines on how the candidate performed and how they can improve.\n"

        f"Transcript: {latest_transcript}\n"
        "Consider the above transcript. It is the self-introduction of the candidate. "
        "If the transcription is null score should be 0.\n"
        "Scoring should be based on the following criteria:\n"
        "1. Whether they introduced themselves.\n"
        "2. Whether they mentioned their studies and experience.\n"
        "Based on that, give a self-introduction score out of 10.\n"
    

        "Output example:\n"
        "{\n"
        '"overall_score": {score out of 10},\n'
        '"self_introduction_score": {score out of 10},\n'
        '"role_play_score": {score out of 10},\n'
        '"feedback": "{feedback in 2 lines consider only the roleplay conversation and give a genuine feedback on how he initiated the call and in what areas he need to improve while building the conversation with the customer.}"\n'
        "}"
    )

    try:
        # Send the prompt to the AI model
        response = chat.send_message(grammar_check_prompt)

        # Print raw response for debugging
        print(f"Raw response: {response.text}")

        # Extract JSON part using regex
        json_match = re.search(r'\{.*?\}', response.text, re.DOTALL)

        if json_match:
            grammar_check_data = json.loads(json_match.group(0))  # Convert JSON string to dictionary
        else:
            print("No valid JSON found in the response.")
            return {"error": "Failed to generate grammar report due to missing JSON data."}

        print(f"Parsed grammar_check_data: {grammar_check_data}")

        # Extract correct keys
        self_introduction_score = grammar_check_data.get("self_introduction_score", 0)
        role_play_score = grammar_check_data.get("role_play_score", 0)
        overall_score = (self_introduction_score + role_play_score)/2
        feedback = grammar_check_data.get("feedback", "The conversation was engaging.")

        # Return structured dictionary
        return {
            "overall_score": overall_score,
            "self_introduction_score": self_introduction_score,
            "role_play_score": role_play_score,
            "feedback": feedback
        }

    except json.JSONDecodeError:
        print("Failed to parse the response as JSON.")
        return {"error": "Failed to generate grammar report due to JSON parsing issue."}

    except Exception as e:
        print(f"Error during grammar check: {e}")
        return {"error": "Failed to generate grammar report due to an unknown error."}




# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

latest_transcript=''
# API to handle audio file upload and save it in the static/audio directory
@app.route('/save_transcript', methods=['POST'])
def save_transcript():
    global latest_transcript
    data = request.get_json()
    if 'transcript' in data:
        latest_transcript = data['transcript']
        print(latest_transcript )
        return jsonify({"message": "Transcript saved successfully"}), 200
    return jsonify({"error": "No transcript provided"}), 400


@app.route('/get_transcript', methods=['GET'])
def get_transcript():
    return jsonify({"transcript": latest_transcript}), 200

# Route for Task 2
@app.route('/q2')
def q2():
    return render_template('q2.html')

@app.route('/index1')
def index1():
    return render_template('index1.html')

chat_history = [] 

@app.route('/api', methods=['POST'])
def chat_api():
    global last_message_time, grammar_report

    # if conversation_ended:
    #     return jsonify({"response": "Your interview has already ended.", "end": True})

    user_message = request.json.get('message', '').strip()

    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    # Store the user message in chat history
    chat_history.append({"role": "user", "message": user_message})

    # Reset the last message time on receiving a message
    last_message_time = time.time()

 

    # Check if the user wants to end the conversation
    end_keywords = ["thank you", "i will get back to you", "goodbye"]
    if any(keyword in user_message.lower() for keyword in end_keywords):
        # conversation_ended = True
        if chat:
            grammar_report = perform_grammar_check()  # Trigger grammar check when user ends the conversation
        return jsonify({
            "response": "Thank you for the conversation. Your interview has ended.",
            "end": True,
            "redirect_url": "/view_report"
        })
     
    # Handle the user's message
    bot_response = handle_message(user_message)
    speak_response(bot_response,1.2)

    return jsonify({
        "response": bot_response,
        "end": False
    })


@app.route('/view_report')
def view_report():
    global grammar_report

    if not grammar_report or "error" in grammar_report:
        grammar_report = perform_grammar_check()  # Generate the report if not available
    chat_history.clear()

    return render_template('r.html', 
                           overall_score=grammar_report.get("overall_score", "N/A"), 
                           selfintroduction_score=grammar_report.get("self_introduction_score", "N/A"), 
                           roleplay_score=grammar_report.get("role_play_score", "N/A"), 
                           feedback=grammar_report.get("feedback", "No feedback available"))


# Route for ending the conversation
@app.route('/end')
def end_conversation():
    return render_template('end1.html')

if __name__ == '__main__':
    start_new_chat_session()  # Initialize the chat session
    # last_message_time = time.time()  # Set initial time
    # # Start the timeout thread
    # timeout_thread = threading.Thread(target=check_timeout)
    # timeout_thread.start()

    app.run()
