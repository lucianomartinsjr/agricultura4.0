from crewai import Agent, Task, Crew, Process, LLM
from langchain_community.tools import DuckDuckGoSearchRun, Tool
from flask import Flask, request, jsonify, render_template
import os
import requests
from dotenv import load_dotenv
import logging
from time import time, sleep

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

def setup_agents(my_llm, tools, tipo_cultura, estagio_cultura, sintomas, ultimos_tratamentos, temperatura, umidade_solo, umidade_ar, local, regiao):
    buscador = Agent(
        role='Agente de Busca e Análise de Cultura',
        goal='Buscar informações específicas sobre a cultura e condições climáticas',
        backstory=f'''
            Você é um especialista em agricultura e meteorologia.
            Quero que forneça informações sobre as condições climáticas atuais e previsões para a cultura de {tipo_cultura} na região {regiao} do Brasil, localização específica: {local}.
            
            Considere:
            - Temperatura atual: {temperatura}°C
            - Umidade do Solo: {umidade_solo}%
            - Umidade do Ar: {umidade_ar}%
            
            FORMATO OBRIGATÓRIO PARA RESPOSTAS:
            1. Para usar uma ferramenta:
                Thought: [seu pensamento]
                Action: duckduckgo_search
                Action Input: {{"query": "sua busca aqui"}}
                Observation: [resultado da busca]
            
            2. Para resposta final:
                Thought: Agora posso fornecer uma resposta completa
                Final Answer: [sua resposta detalhada em português]
            
            Não desvie deste formato exato.
        ''',
        llm=my_llm,
        verbose=True,
        allow_delegation=False,
        tools=tools
    )
    
    fitopatologista = Agent(
        role='Especialista em Diagnóstico e Tratamento',
        goal='Analisar sintomas e recomendar tratamentos específicos',
        backstory=f'''
            Você é um consultor agrícola especializado em fitopatologia.
            
            Baseado na cultura de {tipo_cultura} e nas condições:
            - Estágio: {estagio_cultura}
            - Sintomas: {sintomas}
            - Tratamentos anteriores: {ultimos_tratamentos}
            - Temperatura: {temperatura}°C
            - Umidade do Solo: {umidade_solo}%
            - Umidade do Ar: {umidade_ar}%
            
            Liste as pragas e doenças mais comuns e recomende tratamentos ou práticas preventivas.
            Considere também condições locais e histórico de tratamentos.
            
            FORMATO OBRIGATÓRIO PARA RESPOSTAS:
            1. Para usar uma ferramenta:
                Thought: [seu pensamento]
                Action: duckduckgo_search
                Action Input: {{"query": "sua busca aqui"}}
                Observation: [resultado da busca]
            
            2. Para resposta final:
                Thought: Agora posso fornecer uma resposta completa
                Final Answer: [sua resposta detalhada em português]
            
            Não desvie deste formato exato.
        ''',
        tools=tools,
        llm=my_llm,
        verbose=True,
        allow_delegation=False
    )
    
    especialista_insumos = Agent(
        role='Especialista em Insumos e Equipamentos',
        goal='Recomendar produtos e equipamentos específicos para a cultura',
        backstory=f'''
            Você é um especialista em insumos agrícolas e equipamentos.
            Para a cultura de {tipo_cultura} no estágio {estagio_cultura}, considerando:
            - Temperatura: {temperatura}°C
            - Umidade do Solo: {umidade_solo}%
            - Umidade do Ar: {umidade_ar}%
            
            Recomende fertilizantes, pesticidas e equipamentos adequados.
            Inclua detalhes como frequência de uso, quantidade recomendada e especificações técnicas.
            
            FORMATO OBRIGATÓRIO PARA RESPOSTAS:
            1. Para usar uma ferramenta:
                Thought: [seu pensamento]
                Action: duckduckgo_search
                Action Input: {{"query": "sua busca aqui"}}
                Observation: [resultado da busca]
            
            2. Para resposta final:
                Thought: Agora posso fornecer uma resposta completa
                Final Answer: [sua resposta detalhada em português]
            
            Não desvie deste formato exato.
        ''',
        tools=tools,
        llm=my_llm,
        verbose=True,
        allow_delegation=False
    )
    
    agente_monitoramento = Agent(
        role='Agente de Monitoramento',
        goal='Monitorar dados dos sensores e recomendar ações',
        backstory=f'''
            Você é um sistema de monitoramento agrícola responsável por interpretar dados de sensores em tempo real.
            
            Para a cultura {tipo_cultura}:
            - Temperatura atual: {temperatura}°C
            - Umidade do Solo: {umidade_solo}%
            - Umidade do Ar: {umidade_ar}%
            - Estágio: {estagio_cultura}
            
            Compare com os parâmetros ideais e determine ações necessárias.
            
            FORMATO OBRIGATÓRIO PARA RESPOSTAS:
            1. Para usar uma ferramenta:
                Thought: [seu pensamento]
                Action: duckduckgo_search
                Action Input: {{"query": "sua busca aqui"}}
                Observation: [resultado da busca]
            
            2. Para resposta final:
                Thought: Agora posso fornecer uma resposta completa
                Final Answer: [sua resposta detalhada em português]
            
            Não desvie deste formato exato.
        ''',
        tools=tools,
        llm=my_llm,
        verbose=True,
        allow_delegation=False
    )
    
    return buscador, fitopatologista, especialista_insumos, agente_monitoramento

def setup_tasks(buscador, fitopatologista, especialista_insumos, agente_monitoramento, tipo_cultura, estagio_cultura, local, regiao):
    tarefa_busca = Task(
        description=f'''
            Responda SEMPRE em português do Brasil:
            1. Busque informações sobre {tipo_cultura} em estágio de {estagio_cultura} na região {regiao} do Brasil, localização específica: {local}
            2. Colete dados sobre temperatura ideal, umidade necessária e outras condições específicas para esta região
            3. Compare as condições atuais com as ideais para esta cultura nesta região
            4. Identifique riscos específicos para este estágio e região
        ''',
        agent=buscador,
        expected_output='Relatório detalhado das condições e requisitos para a cultura especificada'
    )
    
    tarefa_diagnostico = Task(
        description=f'''
            Responda SEMPRE em português do Brasil:
            1. Analise os dados climáticos coletados para {tipo_cultura}
            2. Identifique possíveis pragas e doenças com base nas condições atuais
            3. Avalie os tratamentos anteriores
            4. Recomende tratamentos preventivos e corretivos específicos
        ''',
        agent=fitopatologista,
        expected_output='Lista de pragas/doenças prováveis e recomendações de tratamento'
    )
    
    tarefa_insumos = Task(
        description=f'''
            Responda SEMPRE em português do Brasil:
            1. Com base nos dados climáticos e recomendações do fitopatologista
            2. Sugira fertilizantes e produtos adequados para {tipo_cultura}
            3. Recomende equipamentos necessários para aplicação
            4. Forneça especificações de uso e aplicação
        ''',
        agent=especialista_insumos,
        expected_output='Lista de insumos e equipamentos recomendados com especificações'
    )
    
    tarefa_monitoramento = Task(
        description=f'''
            Responda SEMPRE em português do Brasil:
            1. Registre os dados dos sensores de umidade e temperatura para {tipo_cultura}
            2. Compare com os parâmetros ideais para a cultura
            3. Determine ações necessárias com base nas condições atuais
            4. Indique próximas ações recomendadas
        ''',
        agent=agente_monitoramento,
        expected_output='Registro de dados e recomendações de ações imediatas'
    )
    
    return [tarefa_busca, tarefa_diagnostico, tarefa_insumos, tarefa_monitoramento]

def formatar_resposta(resultado, model):
    try:
        if not resultado or not resultado.tasks_output:
            raise ValueError("Resultado vazio recebido")

        tasks = resultado.tasks_output
        resultados_formatados = {
            "busca_e_analise": tasks[0].raw if len(tasks) > 0 else "Sem dados",
            "diagnostico": tasks[1].raw if len(tasks) > 1 else "Sem dados",
            "insumos": tasks[2].raw if len(tasks) > 2 else "Sem dados",
            "monitoramento": tasks[3].raw if len(tasks) > 3 else "Sem dados"
        }
        
        return {
            "success": True,
            "timestamp": time(),
            "resultado": {
                "analise": {
                    "resumo_geral": "Análise da Cultura e Recomendações",
                    "dados_analisados": {
                        "busca_e_analise": {
                            "titulo": "Análise da Cultura",
                            "conteudo": resultados_formatados["busca_e_analise"]
                        },
                        "diagnostico": {
                            "titulo": "Diagnóstico e Tratamentos",
                            "conteudo": resultados_formatados["diagnostico"]
                        },
                        "insumos": {
                            "titulo": "Recomendações de Insumos",
                            "conteudo": resultados_formatados["insumos"]
                        },
                        "monitoramento": {
                            "titulo": "Monitoramento e Ações",
                            "conteudo": resultados_formatados["monitoramento"]
                        }
                    },
                },
                "metadata": {
                    "versao": "1.0",
                    "modelo_usado": model,
                    "data_analise": time(),
                    "idioma": "pt-BR"
                }
            }
        }
    except Exception as e:
        logger.error(f"Erro ao formatar resposta: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": "Erro ao processar resultados",
            "message": str(e),
            "timestamp": time()
        }

@app.route('/processar', methods=['POST'])
def processar():
    try:
        logger.info("Iniciando processamento de nova requisição")
        dados = request.get_json()
        logger.info(f"Dados recebidos: {dados}")
        
        if 'latitude' in dados and 'longitude' in dados:
            dados['local'] = f"Coordenadas: {dados['latitude']}, {dados['longitude']}"
        
        campos_obrigatorios = {
            'tipo_cultura': 'Tipo da cultura',
            'regiao': 'Região',
            'estagio_cultura': 'Estágio da cultura',
            'sintomas': 'Sintomas observados',
            'ultimos_tratamentos': 'Últimos tratamentos realizados',
            'temperatura': 'Temperatura atual',
            'umidade_solo': 'Umidade do solo',
            'umidade_ar': 'Umidade do ar',
            'local': 'Localização da cultura'
        }
        
        for campo in ['temperatura', 'umidade_solo', 'umidade_ar']:
            if campo in dados and isinstance(dados[campo], str):
                dados[campo] = float(dados[campo])
        
        campos_faltantes = []
        for campo, descricao in campos_obrigatorios.items():
            if campo not in dados or not dados[campo]:
                campos_faltantes.append(descricao)
                logger.warning(f"Campo faltante ou vazio: {campo} ({descricao})")
        
        if campos_faltantes:
            erro_response = {
                "success": False,
                "error": "Campos obrigatórios ausentes",
                "campos_faltantes": campos_faltantes,
                "mensagem": "Por favor, preencha todos os campos obrigatórios"
            }
            logger.error(f"Validação falhou: {erro_response}")
            return jsonify(erro_response), 400

        my_llm, tools = setup_environment()
        
        max_retries = 3
        for tentativa in range(max_retries):
            try:
                # Adicionando tempo de espera entre tentativas
                if tentativa > 0:
                    sleep(tentativa * 2)  # Espera progressiva
                
                # Setup dos agentes
                buscador, fitopatologista, especialista_insumos, agente_monitoramento = setup_agents(
                    my_llm, tools,
                    dados['tipo_cultura'],
                    dados['estagio_cultura'],
                    dados['sintomas'],
                    dados['ultimos_tratamentos'],
                    dados['temperatura'],
                    dados['umidade_solo'],
                    dados['umidade_ar'],
                    dados['local'],
                    dados['regiao']
                )
         
                tarefas = setup_tasks(
                    buscador, 
                    fitopatologista, 
                    especialista_insumos,
                    agente_monitoramento,
                    dados['tipo_cultura'],
                    dados['estagio_cultura'],
                    dados['local'],
                    dados['regiao']
                )
                

                crew = Crew(
                    agents=[buscador, fitopatologista, especialista_insumos, agente_monitoramento],
                    tasks=tarefas,
                    verbose=1,
                    process=Process.sequential
                )
                
                resultados = crew.kickoff()
                
                if not resultados:
                    raise ValueError("Nenhum resultado obtido dos agentes")
                    
                response = formatar_resposta(resultados, my_llm.model)
                if not response.get("success"):
                    return jsonify(response), 500
                    
                return jsonify(response)
                
            except Exception as crew_error:
                logger.warning(f"Tentativa {tentativa + 1} falhou: {str(crew_error)}")
                if tentativa == max_retries - 1:
                    raise Exception(f"Todas as tentativas falharam. Último erro: {str(crew_error)}")
                continue

    except Exception as e:
        logger.error(f"Erro durante o processamento: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "detalhes": "Erro ao processar a requisição",
            "timestamp": time()
        }), 500

if __name__ == '__main__':
    app.run(debug=True)