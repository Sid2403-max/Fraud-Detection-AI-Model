import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Set Page Configuration
st.set_page_config(
    page_title="UPIShield - UPI Fraud Awareness & Risk Simulator",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Inject Premium CSS Styling for a stunning, custom UI
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* 1. Force the entire app background to pure black */
    .stApp {
        background-color: #000000 !important;
    }
    
    /* 2. Turn all standard text, paragraphs, lists, and spans to stark white */
    .stMarkdown p, .stMarkdown li, span, th, td {
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif;
    }
    
    /* 3. Turn all form widget labels (like 'Transaction Type') to bold white */
    div[data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* 4. Ensure headers and subheaders stand out in white */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    /* 5. Clean styling for tabs if you are using them */
    button[data-baseweb="tab"] p {
        color: #a0aec0 !important; /* Muted gray for unselected tabs */
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #ffffff !important; /* Bright white for active tab */
    }
    
    /* Keeping your gorgeous title text white inside the blue banner */
    .blue-banner-title {
        color: #ffffff !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    /* App Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }

    .header-banner p {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.9;
        margin: 0;
        color: #ffffff !important;
    }

    /* Premium Styled Card Component (Glassmorphism Dark) */
    .premium-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }

    /* Mobile Frame Simulated Container */
    .mobile-mockup {
        border: 8px solid #1e293b;
        border-radius: 36px;
        padding: 20px 15px;
        background: #0f172a;
        max-width: 450px;
        margin: 0 auto;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }

    /* Risk Score Styling */
    .risk-badge {
        font-size: 1.4rem;
        font-weight: 700;
        padding: 10px 20px;
        border-radius: 12px;
        text-align: center;
        margin: 15px 0;
        color: white !important;
    }
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    .risk-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }
    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }

    /* Educational Section Styling (Dark Card) */
    .edu-card {
        border-left: 5px solid #3b82f6;
        background: rgba(30, 41, 59, 0.35);
        padding: 15px;
        border-radius: 0 12px 12px 0;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# 2. Load the Saved ML Assets safely
@st.cache_resource
def load_assets():
    model = joblib.load('upi_fraud_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

try:
    model, model_features = load_assets()
except FileNotFoundError:
    st.error("❌ Could not find 'upi_fraud_model.pkl' or 'model_features.pkl'. Please ensure they are in the same folder as this script.")
    st.stop()

# 3. Application Header (SDG 4 Focus - Quality Education on Safe Digital Payments)
st.markdown("""
<div class="header-banner">
    <h1 class="blue-banner-title">🛡️ UPIShield</h1>
    <p>UPI Fraud Awareness & Transaction Risk Simulator</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Welcome to **UPIShield**, an interactive platform developed to educate users on transaction safety. 
Using a Machine Learning model trained on mobile transaction records, we analyze inputs for patterns commonly associated with fraud. 
*Goal: Supporting SDG 4 (Quality Education) by teaching secure digital banking behaviors.*
""")

# Create three tabs: Simulator, Education Hub, and Quiz
tab_sim, tab_edu, tab_quiz = st.tabs(["📱 Risk Simulator", "📚 Education Hub & Tips", "🧠 Cyber Awareness Quiz"])

with tab_sim:
    st.markdown("### 📱 Simulate a Mobile/UPI Transaction")
    st.write("Adjust the transaction details below to see how our AI model assesses risk in real-time.")
    
    # Wrap controls inside a styled container
    with st.container():
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            tx_type = st.selectbox("Transaction Type:", ["TRANSFER", "CASH_OUT"])
            amount = st.number_input("Transaction Amount (₹):", min_value=1.0, value=5000.0, step=500.0)
        with col2:
            old_balance_org = st.number_input("Your Current Balance (₹):", min_value=0.0, value=25000.0, step=1000.0)
            old_balance_dest = st.number_input("Recipient's Est. Balance (₹):", min_value=0.0, value=0.0, step=1000.0)
            
        step = st.slider("Hour of Day (Simulation Step):", min_value=0, max_value=23, value=12, 
                         help="PaySim logs transactions hourly. Simulating hour of the day can affect risk assessments.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # 4. Background Calculations to match PaySim features
    # PaySim needs specific numeric balances after the transaction happens
    new_balance_org = old_balance_org - amount
    new_balance_dest = old_balance_dest + amount

    # 5. Prediction Logic
    if st.button("🔍 Scan Transaction Risk", type="primary", use_container_width=True):
        if amount > old_balance_org:
            st.error("🚨 Transaction Blocked: Insufficient Balance!")
            st.markdown("""
                **Educational Insight:** In standard UPI architecture, your bank container will reject this transaction immediately before it even reaches network verification. 
                
                *However, scammers often try to trick users into executing large transfers by promising 'double returns' or cashbacks. Always keep track of your actual available balance.*
            """)
        else:
            # Recreate the exact feature dictionary your model expects
            input_data = {
                'step': step,
                'amount': amount,
                'oldbalanceOrg': old_balance_org,
                'newbalanceOrig': new_balance_org,
                'oldbalanceDest': old_balance_dest,
                'newbalanceDest': new_balance_dest,
                'type_TRANSFER': 1 if tx_type == "TRANSFER" else 0
            }
            
            # Ensure all trained features exist in our input dataframe, filled with 0 if missing
            input_df = pd.DataFrame([input_data])
            for col in model_features:
                if col not in input_df.columns:
                    input_df[col] = 0.0
                    
            # Reorder columns to match the model training order perfectly
            input_df = input_df[model_features]
            
            # Run prediction probabilities
            try:
                # Check if predict_proba is supported by the loaded model
                probs = model.predict_proba(input_df)[0]
                prediction_prob = probs[1] * 100
            except Exception as e:
                # Fallback or display custom error if predict_proba fails
                st.error(f"Error calling model predictions: {e}")
                st.stop()
            
            # 6. Display Results
            st.markdown("### 📊 Risk Assessment Result")
            
            # We also want to explain what is happening under the hood (SDG 4 Education)
            if prediction_prob < 30:
                st.markdown(f'<div class="risk-badge risk-low">🟢 Low Risk Level: {prediction_prob:.2f}%</div>', unsafe_allow_html=True)
                st.success("✔️ This transaction pattern looks typical and displays low fraud flags.")
                st.info("""
                    💡 **Safe Practice Tip:** Even for low-risk transfers:
                    - Double-check the recipient's name that displays in your UPI app before clicking 'Pay'.
                    - Be cautious of anyone claiming they sent money by mistake and asking you to return it immediately.
                """)
            elif 30 <= prediction_prob < 75:
                st.markdown(f'<div class="risk-badge risk-medium">🟡 Moderate Risk Level: {prediction_prob:.2f}%</div>', unsafe_allow_html=True)
                st.warning("⚠️ This transaction triggers potential risk flags. Please proceed with caution.")
                st.markdown("""
                    <div class="edu-card">
                        <strong>📢 Critical Fraud Prevention Guidance:</strong>
                        <ul>
                            <li><strong>Entering UPI PIN = Sending Money:</strong> Remember, you NEVER need to enter your UPI PIN to <i>receive</i> money. If someone sent you a QR code and asked you to enter your PIN to claim a prize or payment, it is 100% a scam.</li>
                            <li><strong>Request Money Feature:</strong> Scammers often use the 'Request' feature of UPI apps hoping you will tap approve and enter your PIN. Always read the screen carefully.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-badge risk-high">🔴 High Risk Level: {prediction_prob:.2f}%</div>', unsafe_allow_html=True)
                st.error("🚨 Highly suspicious transaction signature detected. High risk of fraud!")
                st.markdown("""
                    <div class="edu-card" style="border-left-color: #ef4444; background: #fef2f2;">
                        <strong>🛑 Stop Immediately & Read:</strong>
                        <ul>
                            <li><strong>Part-Time Job / Task Scams:</strong> Were you asked to perform simple online tasks (like liking videos) and then told to pay deposit money to unlock earnings? This is a classic prepaid task scam.</li>
                            <li><strong>Urgency & Panic Tactics:</strong> Is the caller claiming to be a relative in distress, a government official, or customer support demanding immediate payment? Hang up and contact the actual organisation or relative on a known number.</li>
                            <li><strong>Mule Accounts:</strong> The model flags transfers where recipient account balances have sudden, massive updates or go to empty accounts. Do not let strangers use your account to receive funds.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                
            # Explaining the ML logic to the user (Quality Education)
            with st.expander("🔬 How did the AI calculate this risk?"):
                st.write(f"""
                    When you click calculate, the system transforms your inputs into the mathematical features used to train the machine learning model.
                    - **Balance Check:** The model observed your balance dropping by ₹{amount:,.2f} while the recipient's balance increased by ₹{amount:,.2f}. 
                    - **Transaction Signature:** Fraudulent transactions in the historical dataset often drain the entire starting balance of the origin account (`oldbalanceOrg` vs `newbalanceOrig` = 0), and send it to recipient accounts that had a balance of 0 (`oldbalanceDest` = 0).
                    - **Timing:** Simulating this at hour {step} compared to other hours can trigger anomalies depending on normal business hours.
                """)

with tab_edu:
    st.markdown("### 📚 Learn How to Stay Safe Online")
    st.write("Empowering you with knowledge is the best defense against cyber fraud. Review these common vectors.")
    
    # Accordion layout for visual clean aesthetics
    with st.expander("1. ⛔ The 'PIN to Receive' Trap"):
        st.markdown("""
            **The Trap:** A scammer posts an item for sale on Facebook Marketplace or OLX. They contact you saying they want to buy it immediately. They send you a screenshot of a QR code or initiate a transfer, claiming: *"Scan this QR code or enter your PIN to claim the payment."*
            
            **The Reality:** **A PIN is ONLY used to SEND money.** Scanning a QR code or typing your PIN will immediately debit money *from* your account.
        """)
        
    with st.expander("2. 📱 Fake Customer Support & APK Scams"):
        st.markdown("""
            **The Trap:** You search Google for a customer care number (airline, bank, courier). You call the number and the agent asks you to download an app like AnyDesk, TeamViewer, or a custom `.apk` file to resolve your refund/issue.
            
            **The Reality:** Real banks will never ask you to install screen-sharing software or run custom files. These tools allow scammers to view your screens, steal OTPs, and capture passwords in real-time.
        """)
        
    with st.expander("3. 🎯 Task-Based Part-Time Job Scams"):
        st.markdown("""
            **The Trap:** You receive a WhatsApp or Telegram message offering high pay for liking YouTube videos or rating hotels. After a few tasks, you are added to a group and asked to "invest" or deposit money to unlock VIP tasks.
            
            **The Reality:** Once you deposit large amounts, the scammers lock your profile, claim you made an error, demand a "withdrawal fee" to get your money back, and eventually disappear.
        """)

    with st.expander("4. 🛡️ What to do if you are scammed"):
        st.markdown("""
            1. **Block Immediately:** Lock your UPI IDs, block the numbers, and notify your bank to freeze transactions.
            2. **Report to Authorities:** In India, dial **1930** (National Cyber Crime Helpline) or file a complaint on [cybercrime.gov.in](https://cybercrime.gov.in).
            3. **Report UPI handles:** Report the merchant or UPI handle inside your app (GPay, PhonePe, Paytm, BHIM) to block the fraudster's wallet.
        """)

with tab_quiz:
    st.subheader("🧠 Test Your Fraud Awareness IQ")
    st.markdown("Go through these real-world scenarios to see if you can spot a UPI scam before it happens.")

    # Setup Session State for Quiz Scoring so the page doesn't reset on every click
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
    if 'answers_submitted' not in st.session_state:
        st.session_state.answers_submitted = {}

    # --- Scenario 1 ---
    st.markdown("### **Scenario 1: The Magic Refund**")
    st.info("💬 You receive a call from an executive claiming they accidentally sent ₹2,000 to your Google Pay account. They ask you to open your app and check your notifications to approve a reverse transfer.")

    q1_choice = st.radio(
        "What should you do?",
        ["Open the app immediately and enter your PIN to return the money.", 
         "Check your bank balance directly without clicking any link or entering a PIN.", 
         "Send them your debit card details instead."],
        key="q1"
    )

    if st.button("Submit Answer for Scenario 1"):
        st.session_state.answers_submitted['q1'] = True
        if q1_choice == "Check your bank balance directly without clicking any link or entering a PIN.":
            st.success("🎯 Correct! Scammers use 'Collect Requests' to trick you into entering your PIN, which *deducts* money from your account instead of receiving it.")
        else:
            st.error("🚨 Dangerous! Never enter your UPI PIN based on a stranger's phone call. You never need a PIN to receive money.")

    # --- Scenario 2 ---
    st.markdown("### **Scenario 2: The APK Customer Care Trick**")
    st.info("💬 Your online delivery is delayed. You look up a helpline number online, and the agent asks you to install a small support app called 'SupportHelp.apk' to track your delivery live.")

    q2_choice = st.radio(
        "What is the risk here?",
        ["It is completely safe as long as it is from a customer support agent.", 
         "The app could be malware or a screen-share tool that captures your UPI PIN while you type it.", 
         "The app will just speed up my delivery process."],
        key="q2"
    )

    if st.button("Submit Answer for Scenario 2"):
        st.session_state.answers_submitted['q2'] = True
        if q2_choice == "The app could be malware or a screen-share tool that captures your UPI PIN while you type it.":
            st.success("🎯 Correct! Installing side-loaded `.apk` files allows fraudsters to mirror your screen and intercept bank OTPs.")
        else:
            st.error("🚨 Incorrect. Never download apps via links or third-party sites sent by 'agents'. Always use official app stores.")

    # Calculate score dynamically
    score = 0
    total = 2
    if st.session_state.answers_submitted.get('q1') and q1_choice == "Check your bank balance directly without clicking any link or entering a PIN.":
        score += 1
    if st.session_state.answers_submitted.get('q2') and q2_choice == "The app could be malware or a screen-share tool that captures your UPI PIN while you type it.":
        score += 1
        
    if len(st.session_state.answers_submitted) > 0:
        st.divider()
        st.markdown(f"### 📊 Your Awareness Score: `{score} / {total}`")
        if score == total:
            st.balloons()
            st.success("🏆 Perfect Score! You have a high Fraud Awareness IQ. Keep following safe digital banking habits!")
        else:
            st.info("💡 Review the tips above to improve your fraud awareness and protect yourself from common scams.")

    # --- Emergency Reporting Quick Link ---
    st.divider()
    st.subheader("🚨 Emergency Action Protocol")
    st.error("""
        **If you or someone you know has been defrauded:**
        1. Call the National Cyber Crime Helpline immediately at **1930**.
        2. File a formal complaint online at [cybercrime.gov.in](https://www.cybercrime.gov.in).
        3. Contact your bank instantly to freeze the target UPI ID / account.
    """)

st.divider()
st.caption("Developed for Social Internship Program | Mapped to SDG 4: Quality Education")
