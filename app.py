from src.predict import predict_spam

# Test messages
while True:
    message = input("Enter a message (or type 'exit'): ")
    if message.lower() == 'exit':
        break
    print("Prediction:", predict_spam(message))
