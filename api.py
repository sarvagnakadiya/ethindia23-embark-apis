from flask import Flask, request, jsonify
import openai
from lighthouseweb3 import Lighthouse
import os
from dotenv import load_dotenv
import io
import speech_recognition as sr
from pydub import AudioSegment
import requests
# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Set your Lighthouse API token
lighthouse_api_token = os.environ.get('LIGHTHOUSE_API_TOKEN')
lh = Lighthouse(token=lighthouse_api_token)


def generate_summary(transcript):
    # Append "please make a summary of it" to the transcript
    prompt = transcript + "\n\n what is this video for"

    # Use OpenAI GPT-3 API to generate a summary
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,  # Adjust the max_tokens parameter based on your preference
        temperature=0.6  # Adjust the temperature parameter based on your preference
    )

    summary = response.choices[0].text.strip()
    return summary



def transcribe_video_from_url(video_url):
    # Download the video from the URL
    video_data = requests.get(video_url).content

    # Save the video data to a file
    video_file_path = "video.mp4"
    with open(video_file_path, "wb") as video_file:
        video_file.write(video_data)

    # Convert the video to WAV format
    video = AudioSegment.from_file(video_file_path, format="mp4")
    audio = video.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    audio.export("audio.wav", format="wav")

    # Initialize recognizer class (for recognizing the speech)
    r = sr.Recognizer()

    # Open the audio file
    with sr.AudioFile("audio.wav") as source:
        audio_text = r.record(source)

    # Recognize the speech in the audio
    text = r.recognize_google(audio_text, language='en-US')

    # Remove the temporary files
    os.remove(video_file_path)
    os.remove("audio.wav")

    # Return the transcribed text
    return text

def save_transcript_to_file(video_url, transcript):
    # Save the transcript as a text file locally
    transcript_filename = f"{video_url.replace('/', '_')}_transcript.txt"
    with open(transcript_filename, 'w') as file:
        file.write(transcript)
    return transcript_filename

@app.route('/api/summarize', methods=['POST'])
def summarize_video():
    # Get the video URL from the request
    video_url = request.json.get('video_url')

    # Transcribe the video
    transcript = transcribe_video_from_url(video_url)

    # Save the transcript as a local file
    transcript_filename = save_transcript_to_file(video_url, transcript)

    # Upload the transcript file to Lighthouse
    upload = lh.upload(source=transcript_filename)

    # Generate a summary using OpenAI GPT-3
    summary = generate_summary(transcript)

    # Remove the local transcript file
    os.remove(transcript_filename)

    # Return the original transcript and the generated summary as JSON
    return jsonify({
        'video_transcript': transcript,
        # 'generated_summary': summary,
        # 'transcript_filename': transcript_filename
    })

if __name__ == '__main__':
    app.run(debug=True)

