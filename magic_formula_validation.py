import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MagicFormulaValidator:
    def __init__(self):
        """Inicializa o validador da Magic Formula"""
        self.benchmark_tickers = {
            'IBOV': '^BVSP',      # Ibovespa
            'IFIX': 'IFIX11.SA',  # Índice de Fundos Imobiliários
            'SMAL': 'SMAL11.SA'   # Small Caps
        }
    
    def get_benchmark_data(self, start_date, end_date):
        """
        Busca dados de benchmarks para comparação
        
        Args:
            start_date (str): Data inicial
            end_date (str): Data final
            
        Returns:
            pd.DataFrame: Preços dos benchmarks
        """
       
        
        benchmark_data = {}
        
        for name, ticker in self.benchmark_tickers.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    benchmark_data[name] = data['Adj Close']
                  
             
                   
            except Exception as e:
                print(f"Erro ao buscar {name}: {e}")
        
        if benchmark_data:
            return pd.DataFrame(benchmark_data)
        else:
            return pd.DataFrame()
    
    def calculate_portfolio_metrics(self, returns_series):
        """
        Calcula métricas de performance do portfólio
        
        Args:
            returns_series (pd.Series): Série de retornos
            
        Returns:
            dict: Métricas calculadas
        """
        if returns_series.empty or returns_series.isna().all():
            return {}
        
        # Remove NaN values
        clean_returns = returns_series.dropna()
        
        if len(clean_returns) < 2:
            return {}
        
        # Retorno total
        total_return = (clean_returns.iloc[-1] / clean_returns.iloc[0]) - 1
        
        # Retornos diários
        daily_returns = clean_returns.pct_change().dropna()
        
        if daily_returns.empty:
            return {'total_return': total_return}
        
        # Métricas de risco
        volatility = daily_returns.std() * np.sqrt(252)  # Anualizada
        sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Drawdown máximo
        running_max = clean_returns.expanding().max()
        drawdown = (clean_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_observations': len(clean_returns)
        }
    
    def create_comparison_chart(self, magic_formula_data, benchmark_data, save_path=None):
        """
        Cria gráfico comparativo entre Magic Formula e benchmarks
        
        Args:
            magic_formula_data (pd.DataFrame): Dados das ações da Magic Formula
            benchmark_data (pd.DataFrame): Dados dos benchmarks
            save_path (str): Caminho para salvar o gráfico
        """
        plt.figure(figsize=(15, 12))
        
        # Subplot 1: Performance Normalizada
        plt.subplot(2, 1, 1)
        
        # Calcular portfólio equally-weighted da Magic Formula
        if not magic_formula_data.empty:
            portfolio_performance = magic_formula_data.mean(axis=1)
            portfolio_normalized = (portfolio_performance / portfolio_performance.iloc[0]) * 100
            plt.plot(portfolio_normalized.index, portfolio_normalized.values, 
                    linewidth=3, label='Magic Formula Portfolio', color='red')
        
        # Benchmarks
        if not benchmark_data.empty:
            for col in benchmark_data.columns:
                benchmark_normalized = (benchmark_data[col] / benchmark_data[col].iloc[0]) * 100
                plt.plot(benchmark_normalized.index, benchmark_normalized.values, 
                        linewidth=2, label=col, alpha=0.8)
        
        plt.axhline(y=100, color='black', linestyle='--', alpha=0.5)
        plt.title('Comparação de Performance: Magic Formula vs Benchmarks\n(Jan 2024 - Dez 2024)', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('Performance Normalizada (Base 100)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Ações Individuais da Magic Formula
        plt.subplot(2, 1, 2)
        
        if not magic_formula_data.empty:
            for col in magic_formula_data.columns:
                stock_normalized = (magic_formula_data[col] / magic_formula_data[col].iloc[0]) * 100
                plt.plot(stock_normalized.index, stock_normalized.values, 
                        linewidth=1.5, label=col, alpha=0.7)
        
        plt.axhline(y=100, color='black', linestyle='--', alpha=0.5)
        plt.title('Performance Individual das Ações da Magic Formula', fontsize=14, fontweight='bold')
        plt.xlabel('Data')
        plt.ylabel('Performance Normalizada (Base 100)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico comparativo salvo em: {save_path}")
        
        plt.show()
    
    def analyze_sector_distribution(self, top_stocks_df):
        """
        Analisa distribuição setorial das ações selecionadas
        
        Args:
            top_stocks_df (pd.DataFrame): DataFrame com as top ações
        """
        plt.figure(figsize=(12, 8))
        
        # Distribuição por setor
        sector_counts = top_stocks_df['setor'].value_counts()
        
        plt.subplot(2, 2, 1)
        sector_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Distribuição Setorial - Top 10 Magic Formula')
        plt.ylabel('')
        
        # Distribuição de Earning Yield
        plt.subplot(2, 2, 2)
        plt.hist(top_stocks_df['earning_yield'], bins=8, alpha=0.7, color='skyblue')
        plt.title('Distribuição de Earning Yield')
        plt.xlabel('Earning Yield')
        plt.ylabel('Frequência')
        
        # Distribuição de ROIC
        plt.subplot(2, 2, 3)
        plt.hist(top_stocks_df['roic'], bins=8, alpha=0.7, color='lightgreen')
        plt.title('Distribuição de ROIC')
        plt.xlabel('ROIC')
        plt.ylabel('Frequência')
        
        # Scatter plot EY vs ROIC
        plt.subplot(2, 2, 4)
        plt.scatter(top_stocks_df['earning_yield'], top_stocks_df['roic'], 
                   alpha=0.7, s=100, color='coral')
        plt.xlabel('Earning Yield')
        plt.ylabel('ROIC')
        plt.title('Earning Yield vs ROIC')
        
        # Adicionar labels dos tickers
        for i, row in top_stocks_df.iterrows():
            plt.annotate(row['ticker'], 
                        (row['earning_yield'], row['roic']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('magic_formula_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_performance_report(self, magic_formula_data, benchmark_data, top_stocks_df):
        """
        Gera relatório detalhado de performance
        
        Args:
            magic_formula_data (pd.DataFrame): Dados das ações da Magic Formula
            benchmark_data (pd.DataFrame): Dados dos benchmarks
            top_stocks_df (pd.DataFrame): DataFrame com as top ações
        """
        
        
        # Métricas do portfólio Magic Formula
        if not magic_formula_data.empty:
            portfolio_series = magic_formula_data.mean(axis=1)
            portfolio_metrics = self.calculate_portfolio_metrics(portfolio_series)
          
        
        
        
        if not benchmark_data.empty:
            for benchmark in benchmark_data.columns:
                benchmark_metrics = self.calculate_portfolio_metrics(benchmark_data[benchmark])
                
        
       
        
        if not magic_formula_data.empty:
            for ticker in magic_formula_data.columns:
                if ticker in top_stocks_df['ticker'].values:
                    stock_metrics = self.calculate_portfolio_metrics(magic_formula_data[ticker])
                  
        
        # Correlações
        if not magic_formula_data.empty and len(magic_formula_data.columns) > 1:
           
            # Calcular retornos diários
            daily_returns = magic_formula_data.pct_change().dropna()
            
            if not daily_returns.empty:
                # Pegar apenas as top 5 para não ficar muito poluído
                top_5_tickers = top_stocks_df['ticker'].head(5).tolist()
                available_tickers = [t for t in top_5_tickers if t in daily_returns.columns]
                
                if len(available_tickers) > 1:
                    correlation_matrix = daily_returns[available_tickers].corr()
                   

def main_validation():
    """Função principal para validação da Magic Formula"""
    
    # Configurações
    DATA_INICIO = "2024-01-02"
    DATA_FIM = "2024-12-30"
    
    
    
    # Carregar resultados da Magic Formula (assume que já foi executado)
    try:
        top_stocks_df = pd.read_csv('magic_formula_top10_2024.csv')
        tickers = top_stocks_df['ticker'].tolist()
        print(f"✅ Carregados {len(tickers)} tickers da Magic Formula")
    except FileNotFoundError:
        print("❌ Erro: Arquivo 'magic_formula_top10_2024.csv' não encontrado")
        return