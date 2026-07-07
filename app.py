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

# --- Language Selection & Localization ---
language = st.sidebar.selectbox("🌐 Choose Language / भाषा चुनें", ["English", "Hindi (हिंदी)"])

# Content Dictionary mapping English keys to Hindi translations
text_db = {
    "English": {
        "title": "🛡️ UPIShield",
        "tagline": "UPI Fraud Awareness & Transaction Risk Simulator",
        "intro": "Welcome to **UPIShield**, an interactive platform developed to educate users on transaction safety. Using a Machine Learning model trained on mobile transaction records, we analyze inputs for patterns commonly associated with fraud. *Goal: Supporting SDG 4 (Quality Education) by teaching secure digital banking behaviors.*",
        "tab_sim": "📱 Risk Simulator",
        "tab_edu": "📚 Education Hub & Tips",
        "tab_quiz": "🧠 Cyber Awareness Quiz",
        
        # Tab 1: Simulator
        "sim_header": "### 📱 Simulate a Mobile/UPI Transaction",
        "sim_desc": "Adjust the transaction details below to see how our AI model assesses risk in real-time.",
        "tx_type": "Select Transaction Type:",
        "tx_transfer": "TRANSFER",
        "tx_cash_out": "CASH_OUT",
        "amount": "Transaction Amount (₹):",
        "balance": "Your Current Balance (₹):",
        "dest_balance": "Recipient's Est. Balance (₹):",
        "hour_slider": "Hour of Day (Simulation Step):",
        "hour_help": "PaySim logs transactions hourly. Simulating hour of the day can affect risk assessments.",
        "scan_btn": "🔍 Scan Transaction Risk",
        "insufficient_balance": "🚨 Transaction Blocked: Insufficient Balance!",
        "insufficient_edu": "**Educational Insight:** In standard UPI architecture, your bank container will reject this transaction immediately before it even reaches network verification.\n\n*However, scammers often try to trick users into executing large transfers by promising 'double returns' or cashbacks. Always keep track of your actual available balance.*",
        "risk_header": "### 📊 Risk Assessment Result",
        "low_risk": "🟢 Low Risk Level",
        "low_risk_desc": "✔️ This transaction pattern looks typical and displays low fraud flags.",
        "low_risk_tip": "💡 **Safe Practice Tip:** Even for low-risk transfers:\n- Double-check the recipient's name that displays in your UPI app before clicking 'Pay'.\n- Be cautious of anyone claiming they sent money by mistake and asking you to return it immediately.",
        "mod_risk": "🟡 Moderate Risk Level",
        "mod_risk_desc": "⚠️ This transaction triggers potential risk flags. Please proceed with caution.",
        "mod_risk_guide": """
<div class="edu-card">
    <strong>📢 Critical Fraud Prevention Guidance:</strong>
    <ul>
        <li><strong>Entering UPI PIN = Sending Money:</strong> Remember, you NEVER need to enter your UPI PIN to <i>receive</i> money. If someone sent you a QR code and asked you to enter your PIN to claim a prize or payment, it is 100% a scam.</li>
        <li><strong>Request Money Feature:</strong> Scammers often use the 'Request' feature of UPI apps hoping you will tap approve and enter your PIN. Always read the screen carefully.</li>
    </ul>
</div>
""",
        "high_risk": "🔴 High Risk Level",
        "high_risk_desc": "🚨 Highly suspicious transaction signature detected. High risk of fraud!",
        "high_risk_guide": """
<div class="edu-card" style="border-left-color: #ef4444; background: #fef2f2;">
    <strong>🛑 Stop Immediately & Read:</strong>
    <ul>
        <li><strong>Part-Time Job / Task Scams:</strong> Were you asked to perform simple online tasks (like liking videos) and then told to pay deposit money to unlock earnings? This is a classic prepaid task scam.</li>
        <li><strong>Urgency & Panic Tactics:</strong> Is the caller claiming to be a relative in distress, a government official, or customer support demanding immediate payment? Hang up and contact the actual organisation or relative on a known number.</li>
        <li><strong>Mule Accounts:</strong> The model flags transfers where recipient account balances have sudden, massive updates or go to empty accounts. Do not let strangers use your account to receive funds.</li>
    </ul>
</div>
""",
        "how_ai_works_title": "🔬 How did the AI calculate this risk?",
        "how_ai_works_desc": """
When you click calculate, the system transforms your inputs into the mathematical features used to train the machine learning model.
- **Balance Check:** The model observed your balance dropping by ₹{amount:,.2f} while the recipient's balance increased by ₹{amount:,.2f}. 
- **Transaction Signature:** Fraudulent transactions in the historical dataset often drain the entire starting balance of the origin account (`oldbalanceOrg` vs `newbalanceOrig` = 0), and send it to recipient accounts that had a balance of 0 (`oldbalanceDest` = 0).
- **Timing:** Simulating this at hour {step} compared to other hours can trigger anomalies depending on normal business hours.
""",

        # Tab 2: Education Hub
        "edu_header": "### 📚 Learn How to Stay Safe Online",
        "edu_desc": "Empowering you with knowledge is the best defense against cyber fraud. Review these common vectors.",
        "edu_exp1_title": "1. ⛔ The 'PIN to Receive' Trap",
        "edu_exp1_desc": "**The Trap:** A scammer posts an item for sale on Facebook Marketplace or OLX. They contact you saying they want to buy it immediately. They send you a screenshot of a QR code or initiate a transfer, claiming: *\"Scan this QR code or enter your PIN to claim the payment.\"*\n\n**The Reality:** **A PIN is ONLY used to SEND money.** Scanning a QR code or typing your PIN will immediately debit money *from* your account.",
        "edu_exp2_title": "2. 📱 Fake Customer Support & APK Scams",
        "edu_exp2_desc": "**The Trap:** You search Google for a customer care number (airline, bank, courier). You call the number and the agent asks you to download an app like AnyDesk, TeamViewer, or a custom `.apk` file to resolve your refund/issue.\n\n**The Reality:** Real banks will never ask you to install screen-sharing software or run custom files. These tools allow scammers to view your screens, steal OTPs, and capture passwords in real-time.",
        "edu_exp3_title": "3. 🎯 Task-Based Part-Time Job Scams",
        "edu_exp3_desc": "**The Trap:** You receive a WhatsApp or Telegram message offering high pay for liking YouTube videos or rating hotels. After a few tasks, you are added to a group and asked to \"invest\" or deposit money to unlock VIP tasks.\n\n**The Reality:** Once you deposit large amounts, the scammers lock your profile, claim you made an error, demand a \"withdrawal fee\" to get your money back, and eventually disappear.",
        "edu_exp4_title": "4. 🛡️ What to do if you are scammed",
        "edu_exp4_desc": "1. **Block Immediately:** Lock your UPI IDs, block the numbers, and notify your bank to freeze transactions.\n2. **Report to Authorities:** In India, dial **1930** (National Cyber Crime Helpline) or file a complaint on [cybercrime.gov.in](https://cybercrime.gov.in).\n3. **Report UPI handles:** Report the merchant or UPI handle inside your app (GPay, PhonePe, Paytm, BHIM) to block the fraudster's wallet.",

        # Tab 3: Quiz
        "quiz_header": "🧠 Test Your Fraud Awareness IQ",
        "quiz_desc": "Go through these real-world scenarios to see if you can spot a UPI scam before it happens.",
        "quiz_q1_title": "### **Scenario 1: The Magic Refund**",
        "quiz_q1_info": "💬 You receive a call from an executive claiming they accidentally sent ₹2,000 to your Google Pay account. They ask you to open your app and check your notifications to approve a reverse transfer.",
        "quiz_q1_label": "What should you do?",
        "quiz_q1_opt1": "Open the app immediately and enter your PIN to return the money.",
        "quiz_q1_opt2": "Check your bank balance directly without clicking any link or entering a PIN.",
        "quiz_q1_opt3": "Send them your debit card details instead.",
        "quiz_q1_btn": "Submit Answer for Scenario 1",
        "quiz_q1_correct": "🎯 Correct! Scammers use 'Collect Requests' to trick you into entering your PIN, which *deducts* money from your account instead of receiving it.",
        "quiz_q1_incorrect": "🚨 Dangerous! Never enter your UPI PIN based on a stranger's phone call. You never need a PIN to receive money.",
        "quiz_q2_title": "### **Scenario 2: The APK Customer Care Trick**",
        "quiz_q2_info": "💬 Your online delivery is delayed. You look up a helpline number online, and the agent asks you to install a small support app called 'SupportHelp.apk' to track your delivery live.",
        "quiz_q2_label": "What is the risk here?",
        "quiz_q2_opt1": "It is completely safe as long as it is from a customer support agent.",
        "quiz_q2_opt2": "The app could be malware or a screen-share tool that captures your UPI PIN while you type it.",
        "quiz_q2_opt3": "The app will just speed up my delivery process.",
        "quiz_q2_btn": "Submit Answer for Scenario 2",
        "quiz_q2_correct": "🎯 Correct! Installing side-loaded `.apk` files allows fraudsters to mirror your screen and intercept bank OTPs.",
        "quiz_q2_incorrect": "🚨 Incorrect. Never download apps via links or third-party sites sent by 'agents'. Always use official app stores.",
        "quiz_score_title": "### 📊 Your Awareness Score: `{score} / {total}`",
        "quiz_perfect": "🏆 Perfect Score! You have a high Fraud Awareness IQ. Keep following safe digital banking habits!",
        "quiz_less": "💡 Review the tips above to improve your fraud awareness and protect yourself from common scams.",
        "emergency_title": "🚨 Emergency Action Protocol",
        "emergency_desc": """
**If you or someone you know has been defrauded:**
1. Call the National Cyber Crime Helpline immediately at **1930**.
2. File a formal complaint online at [cybercrime.gov.in](https://www.cybercrime.gov.in).
3. Contact your bank instantly to freeze the target UPI ID / account.
""",
        "footer": "Developed for Social Internship Program | Mapped to SDG 4: Quality Education",
    },
    "Hindi (हिंदी)": {
        "title": "🛡️ यूपीआईशील्ड (UPIShield)",
        "tagline": "यूपीआई धोखाधड़ी जागरूकता और लेनदेन जोखिम सिम्युलेटर",
        "intro": "**यूपीआईशील्ड (UPIShield)** में आपका स्वागत है, जो उपयोगकर्ताओं को लेनदेन सुरक्षा पर शिक्षित करने के लिए विकसित एक संवादात्मक मंच है। मोबाइल लेनदेन रिकॉर्ड पर प्रशिक्षित मशीन लर्निंग मॉडल का उपयोग करके, हम धोखाधड़ी से जुड़े सामान्य पैटर्न के लिए इनपुट का विश्लेषण करते हैं। *लक्ष्य: सुरक्षित डिजिटल बैंकिंग व्यवहार सिखाकर SDG 4 (गुणवत्तापूर्ण शिक्षा) का समर्थन करना।*",
        "tab_sim": "📱 जोखिम सिम्युलेटर",
        "tab_edu": "📚 शिक्षा केंद्र और सुझाव",
        "tab_quiz": "🧠 साइबर जागरूकता प्रश्नोत्तरी",
        
        # Tab 1: Simulator
        "sim_header": "### 📱 मोबाइल/यूपीआई लेनदेन का अनुकरण करें",
        "sim_desc": "वास्तविक समय में हमारे एआई मॉडल द्वारा जोखिम का आकलन देखने के लिए नीचे दिए गए लेनदेन विवरण को समायोजित करें।",
        "tx_type": "लेनदेन का प्रकार चुनें:",
        "tx_transfer": "पैसे ट्रांसफर (Transfer)",
        "tx_cash_out": "पैसे निकालना (Cash Out)",
        "amount": "लेनदेन की राशि (₹ में):",
        "balance": "आपका वर्तमान बैंक बैलेंस (₹ में):",
        "dest_balance": "प्राप्तकर्ता का अनुमानित बैलेंस (₹ में):",
        "hour_slider": "दिन का घंटा (सिमुलेशन चरण):",
        "hour_help": "पेसिम (PaySim) लेनदेन को प्रति घंटा लॉग करता है। दिन के समय का अनुकरण करने से जोखिम आकलन प्रभावित हो सकता है।",
        "scan_btn": "🔍 लेनदेन जोखिम की जांच करें",
        "insufficient_balance": "🚨 लेनदेन अवरुद्ध: अपर्याप्त बैंक बैलेंस!",
        "insufficient_edu": "**शैक्षणिक जानकारी:** मानक यूपीआई आर्किटेक्चर में, आपका बैंक नेटवर्क सत्यापन तक पहुंचने से पहले ही इस लेनदेन को तुरंत अस्वीकार कर देगा।\n\n*हालांकि, जालसाज अक्सर उपयोगकर्ताओं को 'दोगुना रिटर्न' या कैशबैक का वादा करके बड़े ट्रांसफर करने के लिए गुमराह करने की कोशिश करते हैं। हमेशा अपने वास्तविक उपलब्ध बैलेंस पर नज़र रखें।*",
        "risk_header": "### 📊 जोखिम मूल्यांकन परिणाम",
        "low_risk": "🟢 कम जोखिम स्तर",
        "low_risk_desc": "✔️ यह लेनदेन पैटर्न सामान्य लग रहा है और इसमें धोखाधड़ी के कम संकेत दिखाई दे रहे हैं।",
        "low_risk_tip": "💡 **सुरक्षित अभ्यास युक्ति:** कम जोखिम वाले ट्रांसफर के लिए भी:\n- 'भुगतान करें' (Pay) पर क्लिक करने से पहले अपने यूपीआई ऐप में दिखने वाले प्राप्तकर्ता के नाम की दोबारा जांच करें।\n- किसी भी ऐसे व्यक्ति से सावधान रहें जो दावा करता है कि उसने गलती से पैसे भेज दिए हैं और आपसे तुरंत वापस करने के लिए कह रहा है।",
        "mod_risk": "🟡 मध्यम जोखिम स्तर",
        "mod_risk_desc": "⚠️ यह लेनदेन संभावित जोखिम संकेतों को ट्रिगर करता. कृपया सावधानी से आगे बढ़ें।",
        "mod_risk_guide": """
<div class="edu-card">
    <strong>📢 महत्वपूर्ण धोखाधड़ी रोकथाम मार्गदर्शन:</strong>
    <ul>
        <li><strong>यूपीआई पिन दर्ज करना = पैसे भेजना:</strong> याद रखें, आपको पैसे <i>प्राप्त करने</i> के लिए कभी भी अपना यूपीआई पिन दर्ज करने की आवश्यकता नहीं होती है। यदि किसी ने आपको क्यूआर कोड भेजा है और इनाम या भुगतान का दावा करने के लिए अपना पिन दर्ज करने को कहा है, तो यह 100% घोटाला है।</li>
        <li><strong>मनी रिक्वेस्ट (Request Money) सुविधा:</strong> जालसाज अक्सर यूपीआई ऐप्स की 'रिक्वेस्ट' सुविधा का उपयोग करते हैं, उम्मीद करते हैं कि आप उसे स्वीकार कर लेंगे और अपना पिन दर्ज कर देंगे। हमेशा स्क्रीन को ध्यान से पढ़ें।</li>
    </ul>
</div>
""",
        "high_risk": "🔴 उच्च जोखिम स्तर",
        "high_risk_desc": "🚨 अत्यधिक संदिग्ध लेनदेन हस्ताक्षर का पता चला। धोखाधड़ी का उच्च जोखिम!",
        "high_risk_guide": """
<div class="edu-card" style="border-left-color: #ef4444; background: #fef2f2;">
    <strong>🛑 तुरंत रुकें और पढ़ें:</strong>
    <ul>
        <li><strong>पार्ट-टाइम जॉब / टास्क घोटाले:</strong> क्या आपसे सरल ऑनलाइन काम (जैसे वीडियो लाइक करना) करने के लिए कहा गया था और फिर कमाई अनलॉक करने के लिए डिपॉजिट मनी जमा करने को कहा गया? यह एक क्लासिक प्रीपेड टास्क घोटाला है।</li>
        <li><strong>जल्दबाजी और घबराहट पैदा करने की रणनीति:</strong> क्या फोन करने वाला व्यक्ति मुसीबत में फंसा हुआ रिश्तेदार, सरकारी अधिकारी या कस्टमर सपोर्ट होने का दावा कर रहा है और तुरंत भुगतान की मांग कर रहा है? फोन काटें और वास्तविक संगठन या रिश्तेदार से ज्ञात नंबर पर संपर्क करें।</li>
        <li><strong>म्यूल अकाउंट्स (Mule Accounts):</strong> मॉडल उन ट्रांसफर को फ्लैग करता है जहां प्राप्तकर्ता के खातों में अचानक, भारी बढ़ोतरी होती है या पैसे खाली खातों में जाते हैं। अजनबियों को फंड प्राप्त करने के लिए अपने खाते का उपयोग न करने दें।</li>
    </ul>
</div>
""",
        "how_ai_works_title": "🔬 एआई ने इस जोखिम की गणना कैसे की?",
        "how_ai_works_desc": """
जब आप गणना पर क्लिक करते हैं, तो सिस्टम आपके इनपुट को मशीन लर्निंग मॉडल को प्रशिक्षित करने के लिए उपयोग की जाने वाली गणितीय विशेषताओं में परिवर्तित कर देता है।
- **बैलेंस चेक:** मॉडल ने देखा कि आपका बैलेंस ₹{amount:,.2f} गिर गया जबकि प्राप्तकर्ता का बैलेंस ₹{amount:,.2f} बढ़ गया।
- **लेनदेन हस्ताक्षर (Signature):** ऐतिहासिक डेटासेट में धोखाधड़ी वाले लेनदेन अक्सर मूल खाते के पूरे शुरुआती बैलेंस को खाली कर देते हैं (`oldbalanceOrg` बनाम `newbalanceOrig` = 0), और प्राप्तकर्ता खातों में भेजते हैं जिनका बैलेंस 0 था (`oldbalanceDest` = 0)।
- **समय निर्धारण:** सामान्य व्यावसायिक घंटों की तुलना में घंटा {step} पर इसका अनुकरण करना विसंगतियों को ट्रिगर कर सकता है।
""",

        # Tab 2: Education Hub
        "edu_header": "### 📚 ऑनलाइन सुरक्षित रहने का तरीका जानें",
        "edu_desc": "ज्ञान के साथ खुद को सशक्त बनाना साइबर धोखाधड़ी के खिलाफ सबसे अच्छा बचाव है। इन सामान्य तरीकों की समीक्षा करें।",
        "edu_exp1_title": "1. ⛔ 'पैसे पाने के लिए पिन' का जाल",
        "edu_exp1_desc": "**जाल:** एक जालसाज फेसबुक मार्केटप्लेस या ओएलएक्स (OLX) पर बिक्री के लिए एक आइटम पोस्ट करता है। वे आपसे संपर्क करके कहते हैं कि वे इसे तुरंत खरीदना चाहते हैं। वे आपको एक क्यूआर कोड का स्क्रीनशॉट भेजते हैं या ट्रांसफर शुरू करते हैं, और दावा करते हैं: *\"भुगतान प्राप्त करने के लिए इस क्यूआर कोड को स्कैन करें या अपना पिन दर्ज करें।\"*\n\n**सच्चाई:** **पिन का उपयोग केवल पैसे भेजने (SEND) के लिए किया जाता है।** क्यूआर कोड को स्कैन करने या अपना पिन टाइप करने से तुरंत आपके खाते से पैसे कट जाएंगे।",
        "edu_exp2_title": "2. 📱 फर्जी कस्टमर सपोर्ट और एपीके (APK) घोटाले",
        "edu_exp2_desc": "**जाल:** आप गूगल पर कस्टमर केयर नंबर (एयरलाइन, बैंक, कूरियर) खोजते हैं। आप नंबर पर कॉल करते हैं और एजेंट आपसे अपनी धनवापसी/समस्या को हल करने के लिए एनीडेस्क (AnyDesk), टीमव्यूअर (TeamViewer) या एक कस्टम `.apk` फ़ाइल डाउनलोड करने के लिए कहता है।\n\n**सच्चाई:** असली बैंक कभी भी आपसे स्क्रीन-शेयरिंग सॉफ़्टवेयर इंस्टॉल करने या कस्टम फ़ाइलें चलाने के लिए नहीं कहेंगे। ये उपकरण जालसाजों को आपकी स्क्रीन देखने, ओटीपी चुराने और वास्तविक समय में पासवर्ड दर्ज करने की अनुमति देते हैं।",
        "edu_exp3_title": "3. 🎯 कार्य-आधारित पार्ट-टाइम नौकरी घोटाले",
        "edu_exp3_desc": "**जाल:** आपको यूट्यूब वीडियो लाइक करने या होटलों को रेटिंग देने के लिए उच्च वेतन की पेशकश करने वाला व्हाट्सएप या टेलीग्राम संदेश प्राप्त होता है। कुछ कार्यों के बाद, आपको एक समूह में जोड़ा जाता है और वीआईपी कार्यों को अनलॉक करने के लिए पैसे जमा (इन्वेस्ट) करने को कहा जाता है।\n\n**सच्चाई:** एक बार जब आप बड़ी रकम जमा कर देते हैं, तो जालसाज आपकी प्रोफ़ाइल को ब्लॉक कर देते हैं, दावा करते हैं कि आपने गलती की है, अपना पैसा वापस पाने के लिए 'निकासी शुल्क' की मांग करते हैं, और अंततः गायब हो जाते।",
        "edu_exp4_title": "4. 🛡️ घोटाला होने पर क्या करें",
        "edu_exp4_desc": "1. **तुरंत ब्लॉक करें:** अपने यूपीआई आईडी ब्लॉक करें, नंबर ब्लॉक करें, और लेनदेन को फ्रीज करने के लिए अपने बैंक को सूचित करें।\n2. **अधिकारियों को रिपोर्ट करें:** भारत में, तुरंत **1930** (राष्ट्रीय साइबर अपराध हेल्पलाइन) डायल करें या [cybercrime.gov.in](https://cybercrime.gov.in) पर शिकायत दर्ज करें।\n3. **यूपीआई हैंडल की रिपोर्ट करें:** जालसाज के वॉलेट को ब्लॉक करने के लिए अपने ऐप (GPay, PhonePe, Paytm, BHIM) के भीतर मर्चेंट या यूपीआई हैंडल की रिपोर्ट करें।",

        # Tab 3: Quiz
        "quiz_header": "🧠 अपनी धोखाधड़ी जागरूकता आईक्यू (IQ) का परीक्षण करें",
        "quiz_desc": "वास्तविक दुनिया के इन परिदृश्यों को देखें और समझें कि क्या आप यूपीआई घोटाले को होने से पहले पहचान सकते हैं।",
        "quiz_q1_title": "### **परिदृश्य 1: जादुई रिफंड**",
        "quiz_q1_info": "💬 आपको एक अधिकारी का फोन आता है जो दावा करता है कि उसने गलती से आपके Google Pay खाते में ₹2,000 भेज दिए हैं। वे आपसे अपना ऐप खोलने और उलटा ट्रांसफर (reverse transfer) स्वीकृत करने के लिए अपने नोटिफिकेशन की जांच करने को कहते हैं।",
        "quiz_q1_label": "आपको क्या करना चाहिए?",
        "quiz_q1_opt1": "तुरंत ऐप खोलें और पैसे वापस करने के लिए अपना पिन दर्ज करें।",
        "quiz_q1_opt2": "बिना किसी लिंक पर क्लिक किए या पिन दर्ज किए सीधे अपना बैंक बैलेंस जांचें।",
        "quiz_q1_opt3": "इसके बजाय उन्हें अपने डेबिट कार्ड का विवरण भेजें।",
        "quiz_q1_btn": "परिदृश्य 1 के लिए उत्तर सबमिट करें",
        "quiz_q1_correct": "🎯 सही! जालसाज आपको अपना पिन दर्ज कराने के लिए 'कलेक्ट रिक्वेस्ट' (Collect Requests) का उपयोग करते हैं, जिससे पैसे प्राप्त होने के बजाय आपके खाते से कट जाते हैं।",
        "quiz_q1_incorrect": "🚨 खतरनाक! किसी अजनबी के फोन कॉल के आधार पर कभी भी अपना यूपीआई पिन दर्ज न करें। पैसे प्राप्त करने के लिए आपको कभी भी पिन की आवश्यकता नहीं होती है।",
        "quiz_q2_title": "### **परिदृश्य 2: एपीके (APK) कस्टमर केयर ट्रिक**",
        "quiz_q2_info": "💬 आपकी ऑनलाइन डिलीवरी में देरी हो रही है। आप ऑनलाइन हेल्पलाइन नंबर खोजते हैं, और एजेंट आपसे आपकी डिलीवरी को लाइव ट्रैक करने के लिए 'SupportHelp.apk' नामक एक छोटा सपोर्ट ऐप इंस्टॉल करने के लिए कहता है।",
        "quiz_q2_label": "यहां क्या जोखिम है?",
        "quiz_q2_opt1": "जब तक यह कस्टमर सपोर्ट एजेंट की तरफ से है, यह पूरी तरह से सुरक्षित है।",
        "quiz_q2_opt2": "ऐप मैलवेयर या स्क्रीन-शेयर टूल हो सकता है जो आपके टाइप करते समय आपके यूपीआई पिन को कैप्चर कर लेता है।",
        "quiz_q2_opt3": "ऐप सिर्फ मेरी डिलीवरी प्रक्रिया को तेज करेगा।",
        "quiz_q2_btn": "परिदृश्य 2 के लिए उत्तर सबमिट करें",
        "quiz_q2_correct": "🎯 सही! अनधिकृत (side-loaded) `.apk` फ़ाइलें इंस्टॉल करने से जालसाजों को आपकी स्क्रीन देखने और बैंक ओटीपी को इंटरसेप्ट करने की अनुमति मिलती है।",
        "quiz_q2_incorrect": "🚨 गलत। 'एजेंटों' द्वारा भेजे गए लिंक या तीसरे पक्ष के साइटों के माध्यम से कभी भी ऐप डाउनलोड न करें। हमेशा आधिकारिक ऐप स्टोर का उपयोग करें।",
        "quiz_score_title": "### 📊 आपका जागरूकता स्कोर: `{score} / {total}`",
        "quiz_perfect": "🏆 पूर्ण स्कोर! आपका धोखाधड़ी जागरूकता आईक्यू बहुत उच्च है। सुरक्षित डिजिटल बैंकिंग की आदतों का पालन करना जारी रखें!",
        "quiz_less": "💡 सामान्य घोटालों से खुद को बचाने और अपनी धोखाधड़ी जागरूकता में सुधार करने के लिए ऊपर दिए गए सुझावों की समीक्षा करें।",
        "emergency_title": "🚨 आपातकालीन कार्रवाई प्रोटोकॉल",
        "emergency_desc": """
**यदि आपके या आपके किसी परिचित के साथ धोखाधड़ी हुई है:**
1. राष्ट्रीय साइबर अपराध हेल्पलाइन को तुरंत **1930** पर कॉल करें।
2. [cybercrime.gov.in](https://www.cybercrime.gov.in) पर ऑनलाइन औपचारिक शिकायत दर्ज करें।
3. लक्षित यूपीआई आईडी / खाते को फ्रीज करने के लिए तुरंत अपने बैंक से संपर्क करें।
""",
        "footer": "सामाजिक इंटर्नशिप कार्यक्रम के लिए विकसित | SDG 4: गुणवत्तापूर्ण शिक्षा के तहत मैप किया गया",
    }
}

# Fetch the active language dictionary based on user selection
lang_set = text_db[language]


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
st.markdown(f"""
<div class="header-banner">
    <h1 class="blue-banner-title">{lang_set["title"]}</h1>
    <p>{lang_set["tagline"]}</p>
</div>
""", unsafe_allow_html=True)

st.markdown(lang_set["intro"])

# Create three tabs: Simulator, Education Hub, and Quiz
tab_sim, tab_edu, tab_quiz = st.tabs([lang_set["tab_sim"], lang_set["tab_edu"], lang_set["tab_quiz"]])

with tab_sim:
    st.markdown(lang_set["sim_header"])
    st.write(lang_set["sim_desc"])
    
    # Wrap controls inside a styled container
    with st.container():
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            tx_type_display = st.selectbox(lang_set["tx_type"], [lang_set["tx_transfer"], lang_set["tx_cash_out"]])
            tx_type = "TRANSFER" if tx_type_display == lang_set["tx_transfer"] else "CASH_OUT"
            amount = st.number_input(lang_set["amount"], min_value=1.0, value=5000.0, step=500.0)
        with col2:
            old_balance_org = st.number_input(lang_set["balance"], min_value=0.0, value=25000.0, step=1000.0)
            old_balance_dest = st.number_input(lang_set["dest_balance"], min_value=0.0, value=0.0, step=1000.0)
            
        step = st.slider(lang_set["hour_slider"], min_value=0, max_value=23, value=12, 
                         help=lang_set["hour_help"])
        
        st.markdown('</div>', unsafe_allow_html=True)

    # 4. Background Calculations to match PaySim features
    # PaySim needs specific numeric balances after the transaction happens
    new_balance_org = old_balance_org - amount
    new_balance_dest = old_balance_dest + amount

    # 5. Prediction Logic
    if st.button(lang_set["scan_btn"], type="primary", use_container_width=True):
        if amount > old_balance_org:
            st.error(lang_set["insufficient_balance"])
            st.markdown(lang_set["insufficient_edu"])
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
            st.markdown(lang_set["risk_header"])
            
            # We also want to explain what is happening under the hood (SDG 4 Education)
            if prediction_prob < 30:
                st.markdown(f'<div class="risk-badge risk-low">{lang_set["low_risk"]}: {prediction_prob:.2f}%</div>', unsafe_allow_html=True)
                st.success(lang_set["low_risk_desc"])
                st.info(lang_set["low_risk_tip"])
            elif 30 <= prediction_prob < 75:
                st.markdown(f'<div class="risk-badge risk-medium">{lang_set["mod_risk"]}: {prediction_prob:.2f}%</div>', unsafe_allow_html=True)
                st.warning(lang_set["mod_risk_desc"])
                st.markdown(lang_set["mod_risk_guide"], unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-badge risk-high">{lang_set["high_risk"]}: {prediction_prob:.2f}%</div>', unsafe_allow_html=True)
                st.error(lang_set["high_risk_desc"])
                st.markdown(lang_set["high_risk_guide"], unsafe_allow_html=True)
                
            # Explaining the ML logic to the user (Quality Education)
            with st.expander(lang_set["how_ai_works_title"]):
                st.write(lang_set["how_ai_works_desc"].format(amount=amount, step=step))

with tab_edu:
    st.markdown(lang_set["edu_header"])
    st.write(lang_set["edu_desc"])
    
    # Accordion layout for visual clean aesthetics
    with st.expander(lang_set["edu_exp1_title"]):
        st.markdown(lang_set["edu_exp1_desc"])
        
    with st.expander(lang_set["edu_exp2_title"]):
        st.markdown(lang_set["edu_exp2_desc"])
        
    with st.expander(lang_set["edu_exp3_title"]):
        st.markdown(lang_set["edu_exp3_desc"])

    with st.expander(lang_set["edu_exp4_title"]):
        st.markdown(lang_set["edu_exp4_desc"])

with tab_quiz:
    st.subheader(lang_set["quiz_header"])
    st.markdown(lang_set["quiz_desc"])

    # Setup Session State for Quiz Scoring so the page doesn't reset on every click
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
    if 'answers_submitted' not in st.session_state:
        st.session_state.answers_submitted = {}

    # --- Scenario 1 ---
    st.markdown(lang_set["quiz_q1_title"])
    st.info(lang_set["quiz_q1_info"])

    q1_choice = st.radio(
        lang_set["quiz_q1_label"],
        [lang_set["quiz_q1_opt1"], lang_set["quiz_q1_opt2"], lang_set["quiz_q1_opt3"]],
        key="q1"
    )

    if st.button(lang_set["quiz_q1_btn"]):
        st.session_state.answers_submitted['q1'] = True
        if q1_choice == lang_set["quiz_q1_opt2"]:
            st.success(lang_set["quiz_q1_correct"])
        else:
            st.error(lang_set["quiz_q1_incorrect"])

    # --- Scenario 2 ---
    st.markdown(lang_set["quiz_q2_title"])
    st.info(lang_set["quiz_q2_info"])

    q2_choice = st.radio(
        lang_set["quiz_q2_label"],
        [lang_set["quiz_q2_opt1"], lang_set["quiz_q2_opt2"], lang_set["quiz_q2_opt3"]],
        key="q2"
    )

    if st.button(lang_set["quiz_q2_btn"]):
        st.session_state.answers_submitted['q2'] = True
        if q2_choice == lang_set["quiz_q2_opt2"]:
            st.success(lang_set["quiz_q2_correct"])
        else:
            st.error(lang_set["quiz_q2_incorrect"])

    # Calculate score dynamically
    score = 0
    total = 2
    if st.session_state.answers_submitted.get('q1') and q1_choice == lang_set["quiz_q1_opt2"]:
        score += 1
    if st.session_state.answers_submitted.get('q2') and q2_choice == lang_set["quiz_q2_opt2"]:
        score += 1
        
    if len(st.session_state.answers_submitted) > 0:
        st.divider()
        st.markdown(lang_set["quiz_score_title"].format(score=score, total=total))
        if score == total:
            st.balloons()
            st.success(lang_set["quiz_perfect"])
        else:
            st.info(lang_set["quiz_less"])

    # --- Emergency Reporting Quick Link ---
    st.divider()
    st.subheader(lang_set["emergency_title"])
    st.error(lang_set["emergency_desc"])

st.divider()
st.caption(lang_set["footer"])
