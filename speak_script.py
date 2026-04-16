import sys
import pyttsx3

def speak(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 100)
        voices = engine.getProperty("voices")
        try:
            # Try to use Zira (Index 1)
            engine.setProperty("voice", voices[1].id)
        except IndexError:
            # Fallback to default
            engine.setProperty("voice", voices[0].id)
            
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error in speech process: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text_to_speak = " ".join(sys.argv[1:])
        speak(text_to_speak)
