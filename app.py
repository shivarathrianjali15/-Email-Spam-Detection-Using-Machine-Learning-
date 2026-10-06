import streamlit as st
import joblib

# Load trained model
model = joblib.load("spam_model.pkl")

st.title("📧 Email Spam Detection")

st.write("Enter an email message to check whether it is Spam or Not Spam.")

email = st.text_area("Enter your email message:")

if st.button("Check Email"):
    if email.strip() == "":
        st.warning("Please enter an email message.")
    else:
        prediction = model.predict([email])[0]

        if prediction == "spam":
            st.error("🚨 This email is Spam!")
        else:
            st.success("✅ This email is Not Spam!")