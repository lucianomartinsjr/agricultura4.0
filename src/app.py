from crewai import Agent, Task, Crew, Process, LLM
from langchain_community.tools import DuckDuckGoSearchRun, Tool
from flask import Flask, request, jsonify, render_template
import os
import requests
from dotenv import load_dotenv
import logging
from time import time

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

def setup_environment():
    try:
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY não encontrada nas variáveis de ambiente")
            
        my_llm = LLM(
            model='gemini/gemini-pro',
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.5,
            max_retries=5
        )
        
        search_tool = DuckDuckGoSearchRun()
        
        return my_llm, [search_tool]
    except Exception as e:
        logger.error(f"Erro na configuração do ambiente: {str(e)}", exc_info=True)
        raise Exception(f"Erro ao configurar o ambiente: {str(e)}")

def setup_agents(my_llm, tools, tipo_cultura, estagio_cultura, sintomas, ultimos_tratamentos, temperatura, umidade_solo, umidade_ar):
    buscador = Agent(
        role='Agente de Busca e Análise de Cultura',
        goal='Buscar informações específicas sobre a cultura e suas necessidades baseadas no estágio de desenvolvimento',
        backstory=f'''
            Você é um especialista em agricultura com vasto conhecimento em diferentes culturas.
            Analise as condições ideais para o desenvolvimento da {tipo_cultura} no estágio {estagio_cultura}.
            
            IMPORTANTE: Ao usar ferramentas, siga EXATAMENTE este formato:
            
            Thought: [seu pensamento]
            Action Input: {{"query": "sua busca aqui"}}
            Observation: [resultado da busca]
            
            Final Answer: [sua resposta final em português]
        ''',
        llm=my_llm,
        verbose=True,
        allow_delegation=False,
        tools=tools
    )
    
    fitopatologista = Agent(
        role='Especialista em Diagnóstico e Tratamento',
        goal='Analisar sintomas, histórico de tratamentos e condições ambientais para diagnóstico preciso',
        backstory=f'''
            Você é um fitopatologista especializado em diagnóstico de doenças e pragas.
            
            IMPORTANTE: Ao usar ferramentas, siga EXATAMENTE este formato:
            
            Thought: [seu pensamento]
            Action: duckduckgo_search
            Action Input: {{"query": "sua busca aqui"}}
            Observation: [resultado da busca]
            
            Thought: I now know the final answer
            Final Answer: [sua resposta final em português]
            
            Analise os sintomas relatados: {sintomas}
            Considere os tratamentos anteriores: {ultimos_tratamentos}
            Avalie as condições ambientais atuais:
            - Temperatura: {temperatura}°C
            - Umidade do Solo: {umidade_solo}%
            - Umidade do Ar: {umidade_ar}%
        ''',
        tools=tools,
        llm=my_llm,
        verbose=True,
        allow_delegation=False
    )
    
    especialista_insumos = Agent(
        role='Recomendação de Produtos e Equipamentos Agrícolas',
        goal='Recomendar produtos, fertilizantes, e equipamentos agrícolas com base no tipo de cultura e nas condições do solo/clima.',
        backstory='''
            Você é um especialista em insumos agrícolas e equipamentos.
            Para a cultura de [tipo de cultura], considerando o clima com temperatura de [temperatura] e umidade de [umidade],
            recomende fertilizantes, pesticidas e maquinários adequados. Inclua detalhes como frequência de uso,
            quantidade recomendada e links para produtos, se possível.
        ''',
        tools=tools,
        llm=my_llm,
        verbose=True,
        allow_delegation=False
    )
    
    agente_monitoramento = Agent(
        role='Agente de Coleta de Dados de Sensores',
        goal='Analisar e registrar dados dos sensores e condições da cultura, fornecendo recomendações diretas baseadas nas informações disponíveis',
        backstory=f'''
            Você é um sistema automatizado de monitoramento agrícola especializado em análise de dados.
            Com base nos dados fornecidos:
            - Tipo de cultura: {tipo_cultura}
            - Estágio: {estagio_cultura}
            - Temperatura: {temperatura}°C
            - Umidade do Solo: {umidade_solo}%
            - Umidade do Ar: {umidade_ar}%
            - Sintomas: {sintomas}
            - Tratamentos anteriores: {ultimos_tratamentos}
            
            Forneça uma análise direta das condições e recomendações específicas sem fazer perguntas adicionais.
            Sempre inclua: status atual, riscos identificados e ações recomendadas.
        ''',
        tools=tools,
        llm=my_llm,
        verbose=True,
        allow_delegation=False
    )
    
    return buscador, fitopatologista, especialista_insumos, agente_monitoramento

def formatar_resposta(resultado, model):
    # Garantir que resultado seja sempre uma lista
    resultados = resultado if isinstance(resultado, list) else [resultado]
    
    return {
        "success": True,
        "timestamp": time(),
        "resultado": {
            "analise": {
                "resumo_geral": "Análise da Cultura e Recomendações",
                "dados_analisados": {
                    "busca_e_analise": {
                        "titulo": "Análise da Cultura",
                        "conteudo": resultados[0] if len(resultados) > 0 else ""
                    },
                    "diagnostico": {
                        "titulo": "Diagnóstico e Tratamentos",
                        "conteudo": resultados[1] if len(resultados) > 1 else ""
                    },
                    "insumos": {
                        "titulo": "Recomendações de Insumos",
                        "conteudo": resultados[2] if len(resultados) > 2 else ""
                    },
                    "monitoramento": {
                        "titulo": "Status e Próximas Ações",
                        "conteudo": resultados[3] if len(resultados) > 3 else ""
                    }
                },
                "recomendacoes_gerais": "Com base nas análises acima, recomenda-se atenção aos pontos identificados e seguimento das orientações fornecidas."
            },
            "metadata": {
                "versao": "1.0",
                "modelo_usado": model,
                "data_analise": time(),
                "idioma": "pt-BR"
            }
        }
    }

@app.route('/processar', methods=['POST'])
def processar():
    try:
        logger.info("Iniciando processamento de nova requisição")
        dados = request.get_json()
        
        campos_obrigatorios = ['tipo_cultura', 'estagio_cultura', 'sintomas', 
                             'ultimos_tratamentos', 'temperatura', 'umidade_solo', 'umidade_ar']
        for campo in campos_obrigatorios:
            if campo not in dados:
                raise ValueError(f"Campo obrigatório ausente: {campo}")

        my_llm, tools = setup_environment()
        
        max_retries = 3
        for tentativa in range(max_retries):
            try:
                buscador, fitopatologista, especialista_insumos, agente_monitoramento = setup_agents(
                    my_llm, 
                    tools,
                    dados.get('tipo_cultura'),
                    dados.get('estagio_cultura'),
                    dados.get('sintomas'),
                    dados.get('ultimos_tratamentos'),
                    dados.get('temperatura'),
                    dados.get('umidade_solo'),
                    dados.get('umidade_ar')
                )
                
                tarefa_busca = Task(
                    description=f'''
                        Responda SEMPRE em português do Brasil:
                        1. Busque informações sobre {dados.get('tipo_cultura')} em estágio de {dados.get('estagio_cultura')}
                        2. Compare as condições atuais com as ideais
                        3. Identifique riscos específicos para este estágio
                        4. Forneça recomendações de manejo específicas
                    ''',
                    agent=buscador,
                    expected_output='Relatório detalhado em português das condições e requisitos para a cultura especificada'
                )
                
                tarefa_diagnostico = Task(
                    description=f'''
                        Responda SEMPRE em português do Brasil:
                        1. Analise os sintomas relatados: {dados.get('sintomas')}
                        2. Considere o histórico de tratamentos: {dados.get('ultimos_tratamentos')}
                        3. Avalie se os tratamentos anteriores foram adequados
                        4. Identifique possíveis doenças ou pragas com base nos sintomas
                        5. Sugira tratamentos considerando o histórico e estágio atual
                    ''',
                    agent=fitopatologista,
                    expected_output='Lista em português de possíveis doenças/pragas e recomendações de tratamento'
                )
                
                tarefa_insumos = Task(
                    description=f'''
                        Responda SEMPRE em português do Brasil:
                        1. Analise as necessidades da {dados.get('tipo_cultura')} no estágio {dados.get('estagio_cultura')}
                        2. Recomende insumos e equipamentos adequados considerando as condições atuais
                        3. Forneça especificações de uso e aplicação
                    ''',
                    agent=especialista_insumos,
                    expected_output='Lista em português de insumos e equipamentos recomendados com especificações de uso'
                )
                
                tarefa_monitoramento = Task(
                    description=f'''
                        Responda SEMPRE em português do Brasil:
                        1. Registre e analise as condições atuais:
                           - Status da cultura
                           - Condições ambientais
                           - Sintomas observados
                        2. Identifique riscos com base nos dados disponíveis
                        3. Forneça recomendações diretas de ações necessárias
                        4. Estabeleça prioridades de monitoramento
                        
                        Formato da resposta:
                        - Status Atual: [descrição]
                        - Riscos Identificados: [lista]
                        - Ações Recomendadas: [lista priorizada]
                        - Próximo Monitoramento: [recomendação]
                    ''',
                    agent=agente_monitoramento,
                    expected_output='Relatório estruturado em português com análise e recomendações baseadas nos dados disponíveis'
                )
                
                resultados = []  # Lista para armazenar todos os resultados
                
                # Executar cada tarefa individualmente para coletar todos os resultados
                for tarefa in [tarefa_busca, tarefa_diagnostico, tarefa_insumos, tarefa_monitoramento]:
                    crew = Crew(
                        agents=[buscador, fitopatologista, especialista_insumos, agente_monitoramento],
                        tasks=[tarefa],
                        verbose=1,
                        process=Process.sequential
                    )
                    resultado = crew.kickoff()
                    resultados.append(resultado)
                
                return jsonify(formatar_resposta(resultados, my_llm.model))
            except Exception as crew_error:
                logger.warning(f"Tentativa {tentativa + 1} falhou: {str(crew_error)}")
                if tentativa == max_retries - 1:
                    raise crew_error
                continue

    except Exception as e:
        logger.error(f"Erro durante o processamento: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "detalhes": "Erro ao processar a requisição"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)