from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
import os

class SimpleRAG:
    def __init__(self, google_api_key: str):
        self.google_api_key = google_api_key
        os.environ['GEMINI_API_KEY'] = google_api_key
        
        # Initialize components
        self.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-pro',
            google_api_key=google_api_key
        )
        
        #conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        
        self.prompt = PromptTemplate(
            template="""
            You are a helpful assistant. Answer based on the provided context and conversation history.
            Also explain as deeply as possible, highlighting the important portions too.
            If the content is insufficient, just say you don't know.

            Conversation History:
            {history}

            Context: {context}
            Question: {question}
            """,
            input_variables=['context', 'question', 'history']
        )
    
    def get_youtube_transcript(self, video_id: str) -> str:
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.fetch(video_id, languages=['en'])
            snippets = transcript_list.snippets
            transcript = " ".join(snippet.text for snippet in snippets)
            return transcript
        except Exception as e:
            #Fallback to old method
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                return " ".join(chunk['text'] for chunk in transcript)
            except Exception as e2:
                raise Exception(f"Could not get transcript: {e2}")
    
    def setup_vectorstore(self, text: str):
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        chunks = splitter.create_documents([text])
        
        self.vectorstore = FAISS.from_documents(chunks, self.embedding)
        self.retriever = self.vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 2})
        
        self._setup_chain()
    
    def _setup_chain(self):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        def format_history(history_list):
            if not history_list:
                return "No previous conversation."
            return "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in history_list[-3:]])  # Last 3 exchanges
        
        parallel_chain = RunnableParallel({
            'context': self.retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough(),
            'history': RunnableLambda(lambda x: format_history(self.conversation_history))
        })
        
        parser = StrOutputParser()
        self.chain = parallel_chain | self.prompt | self.llm | parser
    
    def add_to_memory(self, question: str, answer: str):
        self.conversation_history.append({
            'question': question,
            'answer': answer
        })
        
        #Keeping only last 10 exchanges to prevent memory overflow
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def query(self, question: str) -> str:
        if not self.chain:
            return "Please setup the knowledge base first by providing a YouTube video or text."
        
        answer = self.chain.invoke(question)
        self.add_to_memory(question, answer)
        return answer
    
    def clear_memory(self):
        self.conversation_history = []
    
    def get_memory(self) -> List[Dict[str, str]]:
        return self.conversation_history