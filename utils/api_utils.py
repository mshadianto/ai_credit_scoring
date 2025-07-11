"""
API utilities and RAG system for AI Credit Scoring
"""
import requests
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import hashlib
import pandas as pd
from pathlib import Path
import sqlite3
from config.settings import settings

class OpenRouterClient:
    """Client for OpenRouter API with Qwen3 integration"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.base_url = settings.OPENROUTER_BASE_URL
        self.default_model = settings.DEFAULT_MODEL
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://credit-scoring-ai.com",
            "X-Title": "AI Credit Scoring System"
        })
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       model: str = None,
                       max_tokens: int = 1000,
                       temperature: float = 0.3,
                       stream: bool = False) -> Dict[str, Any]:
        """Send chat completion request"""
        
        if not self.api_key:
            return {"error": "API key not provided"}
        
        data = {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"API Error {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.Timeout:
            return {"error": "Request timeout"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        try:
            response = self.session.get(f"{self.base_url}/models")
            if response.status_code == 200:
                return response.json()["data"]
            return []
        except:
            return []

class CreditScoringRAG:
    """RAG (Retrieval Augmented Generation) system for credit scoring"""
    
    def __init__(self, api_key: str = None):
        self.client = OpenRouterClient(api_key)
        self.knowledge_base = self._load_knowledge_base()
        self.conversation_history = []
        self.cache = {}
        
    def _load_knowledge_base(self) -> Dict[str, str]:
        """Load banking and credit scoring knowledge base"""
        return {
            "credit_scoring_basics": """
            Credit scoring adalah sistem penilaian risiko kredit yang menggunakan data historis 
            dan algoritma untuk memprediksi kemungkinan seorang peminjam akan gagal bayar.
            
            Faktor utama dalam credit scoring:
            1. Riwayat pembayaran (35%)
            2. Jumlah utang (30%)
            3. Panjang riwayat kredit (15%)
            4. Jenis kredit (10%)
            5. Kredit baru (10%)
            """,
            
            "indonesia_banking_regulation": """
            Regulasi perbankan Indonesia yang relevan:
            1. PBI No. 23/6/PBI/2021 - Penerapan Manajemen Risiko
            2. POJK No. 40/POJK.03/2019 - Penilaian Kualitas Aset Bank
            3. Basel III Framework - Standar internasional
            4. Surat Edaran BI tentang Credit Scoring
            
            Rasio keuangan penting:
            - LTV (Loan to Value) maksimal 90% untuk KPR
            - DTI (Debt to Income) maksimal 40%
            - NPL (Non Performing Loan) target di bawah 5%
            """,
            
            "risk_assessment": """
            Tingkatan risiko kredit:
            1. Sangat Rendah (0-2%): Excellent credit, suku bunga rendah
            2. Rendah (2-5%): Good credit, suku bunga kompetitif
            3. Sedang (5-10%): Fair credit, suku bunga standar
            4. Tinggi (10-20%): Poor credit, suku bunga tinggi
            5. Sangat Tinggi (>20%): Very poor credit, ditolak
            
            Mitigasi risiko:
            - Agunan (collateral)
            - Asuransi kredit
            - Co-signer
            - Down payment lebih besar
            """,
            
            "machine_learning_credit": """
            Algoritma ML untuk credit scoring:
            1. Logistic Regression - baseline model
            2. Random Forest - ensemble learning
            3. Gradient Boosting - high performance
            4. XGBoost/LightGBM - state-of-the-art
            5. Neural Networks - deep learning
            
            Evaluasi model:
            - AUC-ROC score (target >0.8)
            - Precision dan Recall balance
            - Gini coefficient
            - KS statistic
            - Population Stability Index (PSI)
            """,
            
            "alternative_data": """
            Data alternatif untuk credit scoring:
            1. Data telco - riwayat pembayaran pulsa
            2. Data e-commerce - transaksi online
            3. Data media sosial - behavioral analytics
            4. Data geolocation - pola pergerakan
            5. Data utility bills - listrik, air, gas
            
            Keunggulan:
            - Jangkauan lebih luas (thin file customers)
            - Real-time assessment
            - Deteksi fraud lebih baik
            """,
            
            "regulatory_compliance": """
            Compliance dalam credit scoring:
            1. Fairness - tidak diskriminatif
            2. Transparency - explainable AI
            3. Privacy - perlindungan data pribadi
            4. Accuracy - validasi model berkala
            5. Governance - audit trail lengkap
            
            Best practices:
            - Model validation independent
            - Bias testing regular
            - Documentation lengkap
            - Monitoring performance berkelanjutan
            """
        }
    
    def _get_relevant_context(self, query: str) -> str:
        """Retrieve relevant context from knowledge base"""
        query_lower = query.lower()
        relevant_contexts = []
        
        # Simple keyword matching (in production, use embedding similarity)
        keywords_map = {
            "regulation": ["indonesia_banking_regulation", "regulatory_compliance"],
            "risk": ["risk_assessment"],
            "machine learning": ["machine_learning_credit"],
            "ml": ["machine_learning_credit"],
            "algorithm": ["machine_learning_credit"],
            "model": ["machine_learning_credit"],
            "data": ["alternative_data"],
            "scoring": ["credit_scoring_basics"],
            "credit": ["credit_scoring_basics", "risk_assessment"],
            "compliance": ["regulatory_compliance"],
            "fairness": ["regulatory_compliance"],
            "bias": ["regulatory_compliance"]
        }
        
        for keyword, contexts in keywords_map.items():
            if keyword in query_lower:
                for context in contexts:
                    if context in self.knowledge_base:
                        relevant_contexts.append(self.knowledge_base[context])
        
        # If no specific context found, use general credit scoring info
        if not relevant_contexts:
            relevant_contexts.append(self.knowledge_base["credit_scoring_basics"])
        
        return "\n\n".join(relevant_contexts)
    
    def _generate_cache_key(self, query: str, context: str) -> str:
        """Generate cache key for responses"""
        content = f"{query}_{context}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def query(self, user_query: str, portfolio_context: Dict = None) -> str:
        """Process user query with RAG"""
        
        # Get relevant context
        knowledge_context = self._get_relevant_context(user_query)
        
        # Add portfolio context if provided
        if portfolio_context:
            portfolio_info = f"""
            Portfolio Information:
            - Total Applications: {portfolio_context.get('total_applications', 'N/A')}
            - Approval Rate: {portfolio_context.get('approval_rate', 'N/A')}%
            - Average Credit Score: {portfolio_context.get('average_credit_score', 'N/A')}
            - Default Rate: {portfolio_context.get('default_rate', 'N/A')}%
            """
            knowledge_context = f"{portfolio_info}\n\n{knowledge_context}"
        
        # Check cache
        cache_key = self._generate_cache_key(user_query, knowledge_context)
        if cache_key in self.cache:
            cached_response = self.cache[cache_key]
            if datetime.now() - cached_response['timestamp'] < timedelta(hours=1):
                return cached_response['response']
        
        # Prepare messages
        system_prompt = f"""
        Anda adalah ahli perbankan dan credit scoring dengan pengalaman mendalam di industri keuangan Indonesia.
        Gunakan knowledge base berikut untuk memberikan jawaban yang akurat dan praktis:
        
        {knowledge_context}
        
        Berikan jawaban yang:
        1. Praktis dan actionable
        2. Sesuai dengan regulasi Indonesia
        3. Menggunakan best practices industri
        4. Mudah dipahami namun detail
        5. Mencakup aspek teknis dan bisnis
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        # Add conversation history (last 4 messages)
        if self.conversation_history:
            recent_history = self.conversation_history[-4:]
            messages = [messages[0]] + recent_history + [messages[1]]
        
        # Get response from LLM
        response = self.client.chat_completion(
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        
        if "error" in response:
            return f"Maaf, terjadi kesalahan: {response['error']}"
        
        try:
            ai_response = response['choices'][0]['message']['content']
            
            # Cache the response
            self.cache[cache_key] = {
                'response': ai_response,
                'timestamp': datetime.now()
            }
            
            # Update conversation history
            self.conversation_history.extend([
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": ai_response}
            ])
            
            # Keep only last 10 messages
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return ai_response
            
        except (KeyError, IndexError):
            return "Maaf, tidak dapat memproses respons dari AI."
    
    def analyze_credit_application(self, application_data: Dict) -> str:
        """Analyze specific credit application with AI"""
        
        analysis_prompt = f"""
        Analisis aplikasi kredit berikut dan berikan rekomendasi:
        
        Data Aplikant:
        - Usia: {application_data.get('age')} tahun
        - Pendapatan: Rp {application_data.get('income'):,}
        - Lama Bekerja: {application_data.get('employment_years')} tahun
        - Debt-to-Income: {application_data.get('debt_to_income'):.1%}
        - Riwayat Kredit: {application_data.get('credit_history_length')} tahun
        - Skor Pembayaran: {application_data.get('payment_history'):.1%}
        - Credit Utilization: {application_data.get('credit_utilization'):.1%}
        - Pendidikan: {application_data.get('education_level')}
        - Status Pernikahan: {application_data.get('marital_status')}
        - Kepemilikan Rumah: {application_data.get('home_ownership')}
        
        Berikan analisis komprehensif meliputi:
        1. Penilaian risiko kredit
        2. Faktor positif dan negatif
        3. Rekomendasi keputusan
        4. Saran mitigasi risiko
        5. Estimasi suku bunga yang sesuai
        """
        
        return self.query(analysis_prompt)
    
    def get_credit_insights(self, credit_type: str) -> str:
        """Get insights for specific credit type"""
        
        insights_prompt = f"""
        Berikan insights mendalam tentang {credit_type} meliputi:
        1. Karakteristik unik produk ini
        2. Faktor risiko utama yang perlu diperhatikan
        3. Best practices dalam underwriting
        4. Strategi pricing yang optimal
        5. Trend pasar terkini
        6. Regulatory considerations
        """
        
        return self.query(insights_prompt)
    
    def explain_model_decision(self, prediction_result: Dict) -> str:
        """Explain ML model decision"""
        
        explanation_prompt = f"""
        Jelaskan hasil prediksi model credit scoring berikut:
        
        - Credit Score: {prediction_result.get('credit_score')}
        - Risk Level: {prediction_result.get('risk_level')}
        - Default Probability: {prediction_result.get('risk_probability', 0):.1%}
        - Decision: {prediction_result.get('decision')}
        
        Berikan penjelasan yang mudah dipahami tentang:
        1. Mengapa model memberikan skor ini
        2. Faktor apa yang paling berpengaruh
        3. Bagaimana cara meningkatkan skor
        4. Tingkat confidence prediksi
        5. Rekomendasi untuk nasabah
        """
        
        return self.query(explanation_prompt)

class APIResponseFormatter:
    """Format API responses consistently"""
    
    @staticmethod
    def success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
        """Format successful API response"""
        return {
            "status": "success",
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def error_response(error: str, code: int = 400) -> Dict[str, Any]:
        """Format error API response"""
        return {
            "status": "error",
            "error": error,
            "code": code,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def credit_score_response(result: Dict[str, Any]) -> Dict[str, Any]:
        """Format credit scoring response"""
        return APIResponseFormatter.success_response({
            "credit_assessment": {
                "credit_score": result.get("credit_score"),
                "score_range": "300-850",
                "risk_level": result.get("risk_level"),
                "risk_probability": result.get("risk_probability"),
                "decision": result.get("decision"),
                "interest_rate": result.get("interest_rate"),
                "confidence": result.get("confidence", 0.0)
            },
            "model_details": {
                "model_predictions": result.get("model_predictions", {}),
                "ensemble_method": "weighted_average",
                "prediction_timestamp": datetime.now().isoformat()
            },
            "recommendations": {
                "action": result.get("decision"),
                "next_steps": APIResponseFormatter._get_next_steps(result.get("decision"))
            }
        })
    
    @staticmethod
    def _get_next_steps(decision: str) -> List[str]:
        """Get next steps based on decision"""
        if decision == "Approved":
            return [
                "Proceed with loan documentation",
                "Verify income and employment",
                "Complete final underwriting review"
            ]
        elif decision == "Review":
            return [
                "Request additional documentation",
                "Consider co-signer option",
                "Review with senior underwriter"
            ]
        else:  # Rejected
            return [
                "Provide decline notice to applicant",
                "Offer credit improvement advice",
                "Suggest reapplication timeline"
            ]

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls: int = 100, time_window: int = 3600):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def is_allowed(self) -> bool:
        """Check if call is allowed"""
        now = time.time()
        # Remove old calls outside time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        return False
    
    def get_reset_time(self) -> float:
        """Get time until rate limit resets"""
        if not self.calls:
            return 0
        return self.time_window - (time.time() - self.calls[0])

# Utility functions
def validate_credit_application(data: Dict[str, Any]) -> List[str]:
    """Validate credit application data"""
    errors = []
    
    required_fields = [
        'age', 'income', 'employment_years', 'debt_to_income',
        'credit_history_length', 'payment_history', 'credit_utilization'
    ]
    
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"Field '{field}' is required")
    
    # Validate ranges
    if 'age' in data and (data['age'] < 18 or data['age'] > 100):
        errors.append("Age must be between 18 and 100")
    
    if 'income' in data and data['income'] <= 0:
        errors.append("Income must be positive")
    
    if 'debt_to_income' in data and (data['debt_to_income'] < 0 or data['debt_to_income'] > 1):
        errors.append("Debt-to-income ratio must be between 0 and 1")
    
    return errors

def log_api_request(endpoint: str, data: Dict, response: Dict, execution_time: float):
    """Log API request for monitoring"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'endpoint': endpoint,
        'request_size': len(str(data)),
        'response_size': len(str(response)),
        'execution_time': execution_time,
        'status': response.get('status', 'unknown')
    }
    
    # In production, send to proper logging system
    print(f"API Log: {log_entry}")

# Global instances
rag_system = CreditScoringRAG()
rate_limiter = RateLimiter()
formatter = APIResponseFormatter()