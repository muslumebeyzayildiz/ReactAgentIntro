from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()# .env dosyasındaki ayarları yüklüyoruz

# Bellek kaydı için SQLite kullanılıyor (geçici bellek için ":memory:")
memory = SqliteSaver.from_conn_string(":memory:")

# ------- ARAÇLAR (TOOLS) -------

# Tavily araması aracını tanımlıyoruz (sadece 1 sonuç dönecek şekilde)
search = TavilySearchResults(max_results=1)

# Ajanın kullanabileceği araçları listeye ekliyoruz
tools = [search]

# ------- AJAN TANIMI -------

# ReAct tarzı sohbet ajanı için LangChain hub’dan hazır bir istem (prompt) alıyoruz
prompt = hub.pull("hwchase17/react-chat")

# Kullanılacak dil modelini belirliyoruz (OpenAI API ile)
llm = OpenAI()

# ReAct ajanını oluşturuyoruz
agent = create_react_agent(llm, tools, prompt)

# Ajan yürütücüsü tanımlanıyor, burada araçlar, ajan, ve belleği yönetmek için checkpoint yapısı kullanılıyor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, checkpoint=memory)

# Ajanın her konuşmayı tanıyabilmesi için benzersiz bir "thread_id" veriliyor (örn. bir kullanıcı oturumu gibi düşünebilirsin)
config = {"configurable": {"thread_id": "abc123"}}

# ------- SOHBET BAŞLATMA -------

if __name__ == '__main__':
    chat_history = []  # İnsan ve yapay zekâ arasındaki tüm konuşmaları tutar

    while True:
        user_input = input("> ")  # Kullanıcıdan giriş al
        chat_history.append(f"Human: {user_input}")  # Sohbet geçmişine kullanıcı mesajını ekle

        response = []  # Yapay zekâdan gelen yanıtın parçalarını tutacak

        # Ajanı başlat ve sohbet geçmişini de girişe ekle
        for chunk in agent_executor.stream(
                {
                    "input": user_input,  # Kullanıcının son mesajı
                    "chat_history": "\n".join(chat_history),  # Tüm geçmiş mesajlar
                },
                config  # Thread ID içeren yapı (ajanın belleğini izlemeye yarar)
        ):
            # Yapay zekânın cevabı parça parça (stream) gelirse yazdır
            if 'text' in chunk:
                print(chunk['text'], end='')  # Her parçayı anlık olarak yazdır
                response.append(chunk['text'])  # Ve listeye ekle

        # Yapay zekâ yanıtını sohbet geçmişine ekle
        chat_history.append(f"AI: {''.join(response)}")

        # Yeni bir tur için ayraç bas
        print("\n----")