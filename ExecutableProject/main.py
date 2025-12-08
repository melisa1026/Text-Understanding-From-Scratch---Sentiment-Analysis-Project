import tkinter as tk
from InitModel import init_model, predict_sentiment

def classify():
    text = input_space.get()
    sentiment = predict_sentiment(text, model)
    result.config(text=sentiment)
    input_space.delete(0, tk.END)


# initialize the model
model = init_model()

# create a GUI
window = tk.Tk()
window.title("Character-Level CNN: Text Sentiment Classifier")

window.geometry("500x300+100+50")

instructions = tk.Label(text="Enter a text:")
instructions.pack(pady=(10, 5))

input_space = tk.Entry(width=50)
input_space.pack(pady=5)

finish_button = tk.Button(window, text="Get Sentiment", command=classify)
finish_button.pack(pady=10)

result = tk.Label(text="-")
result.pack(pady=20)

window.mainloop()
