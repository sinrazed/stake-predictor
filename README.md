# Stake ML Predictor — PoC

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Version](https://img.shields.io/badge/version-0.1.0--alpha-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Node.js](https://img.shields.io/badge/Node.js-18.x+-green?logo=node.js)

A **Proof of Concept** for predicting Stake.com game outcomes (**Mines** & **Coinflip**) using a `TensorFlow.js` model and provably fair seed pattern analysis.  

> ⚠ **Educational Use Only**  
> This does **not** break cryptography. Results are guesses from pattern recognition and not reliable for gambling. You will likely lose money if used for real betting.

---

## 🚀 Quick Install

**Requirements:**  
- Node.js 18+  
- npm  

```bash
# 1. Clone repo
git clone https://github.com/your-username/stake-predictor-poc.git
cd stake-predictor-poc

# 2. Install dependencies
npm install
```



🔮 Usage
When you run the predictor, it will ask you for seeds via CLI:
Client Seed → Get from Stake (Game → Fairness → Copy)
Server Seed Hash → Same place as above
Nonce → The number of bets you've made with current seed pair (next bet’s nonce)

Example: Mines Prediction
```bash

node index.js
Select Game: 1 ( 1 for mines 2 for coinflip )
Enter Client Seed: Your_Active_Client_Seed
Enter Server Seed Hash: Hash_Of_The_Current_Server_Seed
Enter Nonce: 0

[ ✓ ] [ ✓ ] [ X ] [ ✓ ] [ ✓ ]
...
Recommendation: Pick ✓, avoid X
```

Example: Coinflip Prediction
```bash
node index.js

Select Game: 2 ( 1 for mine 2 for coinflip )
Enter Client Seed: Your_Active_Client_Seed
Enter Server Seed Hash: Hash_Of_The_Current_Server_Seed
Enter Nonce: 0

Nonce 0: Heads (82%)
Nonce 1: Heads (71%)
...

Example:

```python-repl
Nonce 0: Heads (82%)
Nonce 1: Heads (71%)
```
🛠 Tech
- Node.js
- TensorFlow.js
- crypto (HMAC-SHA256)


📌 Contribute
Fork → Improve → PR. Ideas:

- Larger training dataset
- Add more games (Limbo, Plinko)
- Better visualizations

- 📜 License
MIT — see LICENSE.
