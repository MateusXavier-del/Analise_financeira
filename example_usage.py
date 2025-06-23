
   
#!/usr/bin/env python3
"""
Exemplo prático de uso da Magic Formula para o mercado brasileiro

Este script demonstra como usar o sistema completo da Magic Formula,
desde a coleta de dados até a análise final.

Autor: Magic Formula Brazil
Data: 2024
"""

import os
import sys
from datetime import datetime, timedelta

# Importar nossos módulos (assumindo que estão no mesmo diretório)
try:
    from magic_formula_main import MagicFormulaAnalyzer
    from magic_formula_validation import MagicFormulaValidator
except ImportError:
    print("❌ Erro: Certifique-se de que os arquivos magic_formula_main.py e magic_formula_validation.py estão no mesmo diretório")
    sys.exit(1)

def run_complete_analysis():
    """
    Executa análise completa da Magic Formula
    """
    print("🚀 MAGIC FORMULA BRAZIL - ANÁLISE COMPLETA")
    print("=" * 60)
    
    # CONFIGURAÇÕES
    # ============================================================================
    # IMPORTANTE: Substitua pelo seu token real da API do Laboratório de Finanças
    TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzUzMzAyNDUxLCJpYXQiOjE3NTA3MTA0NTEsImp0aSI6ImQyOTVlMjkzMzc1MTRlODE5MzJkMzY2ODc4ZDAzY2U1IiwidXNlcl9pZCI6NzR9.YjB-VN1AVDnjwx25IvWVs2uDkgg7LRF-QNgwsW1plMk"
    
    # Datas para análise
    DATA_SELECAO = "2024-01-02"      # Data para seleção das ações
    DATA_INICIO = "2024-01-02"       # Início da análise de performance
    DATA_FIM = "2024-12-30"          # Fim da análise de performance
    
    # Parâmetros da estratégia
    TOP_N_STOCKS = 10                # Número de ações a selecionar
    MIN_PRICE = 1.0                  # Preço mínimo (evitar penny stocks)
    
    # ETAPA 1: INICIALIZAÇÃO
    # ============================================================================
    print(f"\n📊 CONFIGURAÇÕES DA ANÁLISE:")
    print(f"   • Data de Seleção: {DATA_SELECAO}")
    print(f"   • Período de Performance: {DATA_INICIO} até {DATA_FIM}")
    print(f"   • Número de Ações: {TOP_N_STOCKS}")
    print(f"   • Preço Mínimo: R$ {MIN_PRICE:.2f}")
    
    # Verificar se o token foi alterado
    if TOKEN == "dkadjkajljksajklajlknvn847824jk":
        print("\n⚠️  ATENÇÃO: Você precisa substituir o TOKEN pela sua chave real da API!")
        print("   Edite a variável TOKEN na linha 28 deste arquivo.")
        
        resposta = input("\nDeseja continuar mesmo assim? (s/N): ").lower()
        if resposta != 's':
            print("Encerrando...")
            return
    
    # Inicializar analisador
    analyzer = MagicFormulaAnalyzer(TOKEN)
    
    # ETAPA 2: COLETA E PROCESSAMENTO DOS DADOS
    # ============================================================================
    
    
    # Buscar dados da API
    df_raw = analyzer.get_data_from_api(DATA_SELECAO)
    
    if df_raw.empty:
        print(" Erro: Não foi possível obter dados da API")
        return
  
    
 
    
    df_filtered = analyzer.apply_magic_formula_filters(df_raw)
    
    if df_filtered.empty:
        print("❌ Erro: Nenhuma ação passou nos filtros")
    
        return
    

    
    # ETAPA 4: CALCULAR MAGIC FORMULA
    # ============================================================================
  
    
    df_ranked = analyzer.calculate_magic_formula(df_filtered)
    
    # Mostrar estatísticas dos indicadores
   
    # ETAPA 5: SELEÇÃO DAS MELHORES AÇÕES
    # ============================================================================
  
    top_stocks = analyzer.get_top_stocks(df_ranked, TOP_N_STOCKS)
    
    # ETAPA 6: ANÁLISE DE PERFORMANCE HISTÓRICA
    # ============================================================================
 
    
    tickers = top_stocks['ticker'].tolist()
    
    # Buscar dados históricos
    prices_df = analyzer.get_stock_performance(tickers, DATA_INICIO, DATA_FIM)
    
    if prices_df.empty:
      
        returns_data = {}
    else:
        # Calcular retornos
        returns_data = analyzer.calculate_returns(prices_df)
        
        # Mostrar resumo dos retornos
        valid_returns = [ret for ret in returns_data.values() if not pd.isna(ret)]
       
           
    
    # ETAPA 7: GERAÇÃO DE GRÁFICOS E RELATÓRIOS
    # ============================================================================
    print(f"\n6️⃣ GERANDO VISUALIZAÇÕES...")
    
    # GRÁFICO PRINCIPAL: Análise Magic Formula usando APENAS dados da API
  
    analyzer.plot_magic_formula_analysis_from_api(top_stocks, 'analise_magic_formula_api_2024.png')
    
    # Tentar buscar dados históricos (opcional - se falhar, continua sem)
  
    tickers = top_stocks['ticker'].tolist()
    
    try:
        prices_df = analyzer.get_stock_performance(tickers, DATA_INICIO, DATA_FIM)
        
        if not prices_df.empty:
         
            analyzer.plot_performance(prices_df, top_stocks, 'magic_formula_performance_2024.png')
            
            # Calcular retornos
            returns_data = analyzer.calculate_returns(prices_df)
            
            # Mostrar resumo dos retornos
            valid_returns = [ret for ret in returns_data.values() if not pd.isna(ret)]
           
              
        else:
           
            returns_data = {}
            
    except Exception as e:
       
        returns_data = {}
        prices_df = pd.DataFrame()
    
    # Salvar resultados em CSV
    if returns_data:
        top_stocks['retorno_2024'] = top_stocks['ticker'].map(returns_data)
    else:
        top_stocks['retorno_2024'] = 'N/A'  # Sem dados históricos
    
    output_file = 'magic_formula_results_2024.csv'
    columns_to_save = ['posicao', 'ticker', 'setor', 'preco', 'earning_yield', 'roic', 
                      'magic_formula_rank', 'retorno_2024']
    
    top_stocks[columns_to_save].to_csv(output_file, index=False)
    
    
    # ETAPA 8: RELATÓRIO FINAL
    # ============================================================================
   
    
    analyzer.generate_report(top_stocks, returns_data)
    
    # ETAPA 9: VALIDAÇÃO E COMPARAÇÃO (OPCIONAL)
    # ============================================================================
   
    
    try:
        validator = MagicFormulaValidator()
        
        # Análise setorial sempre funciona (usa dados da API)
        validator.analyze_sector_distribution(top_stocks)
        
        # Benchmarks só se conseguimos dados históricos
        if not prices_df.empty:
            benchmark_data = validator.get_benchmark_data(DATA_INICIO, DATA_FIM)
            
            if not benchmark_data.empty:
                validator.create_comparison_chart(prices_df, benchmark_data, 
                                                'magic_formula_comparison_2024.png')
                validator.generate_performance_report(prices_df, benchmark_data, top_stocks)
    
          
        
    except Exception as e:
        print(f" Aviso: Erro na análise comparativa: {e}")
      
      
    
    # ETAPA 10: RESUMO FINAL
    # ============================================================================
  
    
    
    
    try:
        validator = MagicFormulaValidator()
        
        # Buscar benchmarks
        benchmark_data = validator.get_benchmark_data(DATA_INICIO, DATA_FIM)
        
        if not benchmark_data.empty and not prices_df.empty:
            # Gerar análises complementares
            validator.analyze_sector_distribution(top_stocks)
            validator.create_comparison_chart(prices_df, benchmark_data, 
                                            'magic_formula_comparison_2024.png')
            validator.generate_performance_report(prices_df, benchmark_data, top_stocks)
        
    except Exception as e:
        print(f"⚠️  Aviso: Erro na análise comparativa: {e}")
        print("   A análise principal foi concluída com sucesso")
    
   

def show_help():
    """Mostra ajuda sobre como usar o sistema"""
    
    print("📚 MAGIC FORMULA BRAZIL - GUIA DE USO")
    print("="*50)
    
    print("\n🎯 O QUE É A MAGIC FORMULA?")
    print("A Magic Formula é uma estratégia de investimento que seleciona ações")
    print("baseada em dois critérios principais:")
    print("• Earning Yield (EY): Retorno dos lucros")
    print("• Return on Invested Capital (ROIC): Eficiência operacional")
    
    print("\n🚀 COMO USAR ESTE SISTEMA:")
    print("1. Obtenha um token da API do Laboratório de Finanças")
    print("2. Substitua o TOKEN no código")
    print("3. Execute: python example_usage.py")
    
    print("\n📊 O QUE O SISTEMA FAZ:")
    print("• Coleta dados fundamentalistas")
    print("• Aplica filtros de qualidade")
    print("• Calcula rankings da Magic Formula")
    print("• Seleciona as melhores ações")
    print("• Analisa performance histórica")
    print("• Gera gráficos e relatórios")
    print("• Compara com benchmarks do mercado")
    
    print("\n📁 ARQUIVOS GERADOS:")
    print("• CSV com resultados detalhados")
    print("• Gráficos de performance")
    print("• Análise setorial")
    print("• Comparação com índices")
    
    print("\n⚠️  IMPORTANTE:")
    print("• Para fins educacionais apenas")
    print("• Não é recomendação de investimento")
    print("• Faça sua própria análise")

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Verificar argumentos da linha de comando
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_help()
    else:
        run_complete_analysis()