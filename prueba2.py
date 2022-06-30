# import library
import speech_recognition as sr

# Initialize recognizer class (for recognizing the speech)

r = sr.Recognizer()

# Reading Microphone as source
# listening the speech and store in audio_text variable

while True:
    with sr.Microphone() as source:
        print("Talk")
        audio_text = r.listen(source)
        print("Time over, thanks")
        # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling

        try:
            # using google speech recognition
            print("Text: " + r.recognize_google(audio_text))
            #text1 = r.recognize_google(audio_text, language="te-IN")
            #print('Converting audio transcripts into text ...')
            #print(text)
            #print(text1)
        except:
            print("Sorry, I did not get that")