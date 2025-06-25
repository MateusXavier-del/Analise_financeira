import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')
import os
import sys
from datetime import datetime, timedelta

# Configurações para gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MagicFormulaAnalyzer:
    def __init__(self, token):
        """
        Inicializa o analisador da Magic Formula
        
        Args:
            token (str): Token de autenticação da API do Laboratório de Finanças
        """
        self.token = token
        self.headers = {'Authorization': f'JWT eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzUzMzAyNDUxLCJpYXQiOjE3NTA3MTA0NTEsImp0aSI6ImQyOTVlMjkzMzc1MTRlODE5MzJkMzY2ODc4ZDAzY2U1IiwidXNlcl9pZCI6NzR9.YjB-VN1AVDnjwx25IvWVs2uDkgg7LRF-QNgwsW1plMk'}
        self.base_url = 'https://laboratoriodefinancas.com/api/v1/planilhao'
        
    def get_data_from_api(self, data_base):
        """
        Busca dados da API do Laboratório de Finanças
        
        Args:
            data_base (str): Data no formato YYYY-MM-DD
            
        Returns:
            pd.DataFrame: DataFrame com os dados das ações
        """
        params = {'data_base': data_base}
        
        try:
            response = requests.get(self.base_url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == '200_OK' and 'dados' in data:
                df = pd.DataFrame(data['dados'])
                print(f"✅ Dados carregados com sucesso: {len(df)} ações encontradas para {data_base}")
                return df
            else:
                print(f"❌ Erro na resposta da API: {data['status']}")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro na requisição: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")
            return pd.DataFrame()
    
    def apply_magic_formula_filters(self, df):
        """
        Aplica filtros básicos antes de calcular a Magic Formula
        
        Args:
            df (pd.DataFrame): DataFrame com dados das ações
            
        Returns:
            pd.DataFrame: DataFrame filtrado
        """
        initial_count = len(df)
        
        # Filtros básicos
        df_filtered = df.copy()
        
        # Remove ações com dados faltantes nos indicadores principais
        df_filtered = df_filtered.dropna(subset=['earning_yield', 'roic', 'enterprise_value', 'preco'])
        
        # Remove ações com Enterprise Value negativo ou muito baixo
        df_filtered = df_filtered[df_filtered['enterprise_value'] > 0]
        
        # Remove ações com preço muito baixo (possíveis penny stocks)
        df_filtered = df_filtered[df_filtered['preco'] >= 1.0]
        
        # Remove ações financeiras e de utilities (opcional, seguindo Greenblatt)
        setores_excluir = ['bancos', 'seguros', 'energia_eletrica', 'agua_saneamento']
        df_filtered = df_filtered[~df_filtered['setor'].isin(setores_excluir)]
        
        print(f"📊 Filtros aplicados: {initial_count} → {len(df_filtered)} ações")
        return df_filtered
    
    def calculate_magic_formula(self, df):
        """
        Calcula a Magic Formula e ranqueia as ações
        
        Args:
            df (pd.DataFrame): DataFrame com dados das ações
            
        Returns:
            pd.DataFrame: DataFrame com rankings da Magic Formula
        """
        df_magic = df.copy()
        
        # Ranking por Earning Yield (maior é melhor)
        df_magic['rank_earning_yield'] = df_magic['earning_yield'].rank(ascending=False)
        
        # Ranking por ROIC (maior é melhor)
        df_magic['rank_roic'] = df_magic['roic'].rank(ascending=False)
        
        # Magic Formula: soma dos rankings (menor é melhor)
        df_magic['magic_formula_rank'] = df_magic['rank_earning_yield'] + df_magic['rank_roic']
        
        # Ordena pelo ranking da Magic Formula
        df_magic = df_magic.sort_values('magic_formula_rank')
        
        # Adiciona posição final
        df_magic['posicao'] = range(1, len(df_magic) + 1)
        
        return df_magic
    
    def get_top_stocks(self, df, top_n=10):
        """
        Seleciona as top N ações pela Magic Formula
        
        Args:
            df (pd.DataFrame): DataFrame com rankings calculados
            top_n (int): Número de ações a selecionar
            
        Returns:
            pd.DataFrame: Top N ações
        """
        top_stocks = df.head(top_n).copy()
        
        print(f"\n🏆 TOP {top_n} AÇÕES PELA MAGIC FORMULA:")
        print("=" * 80)
        
        for i, row in top_stocks.iterrows():
            print(f"{row['posicao']:2d}. {row['ticker']:6s} | "
                  f"EY: {row['earning_yield']:6.1%} | "
                  f"ROIC: {row['roic']:6.1%} | "
                  f"Preço: R$ {row['preco']:6.2f} | "
                  f"Setor: {row['setor']}")
        
        return top_stocks
    
    def get_stock_performance(self, tickers, start_date, end_date):
        """
        Busca performance das ações no Yahoo Finance
        
        Args:
            tickers (list): Lista de tickers
            start_date (str): Data inicial
            end_date (str): Data final
            
        Returns:
            pd.DataFrame: Preços históricos das ações
        """
        print(f"\n📈 Buscando dados históricos de {start_date} até {end_date}...")
        
        # Adiciona .SA para ações brasileiras
        yahoo_tickers = [ticker + '.SA' for ticker in tickers]
        
        try:
            # Busca dados históricos
            data = yf.download(yahoo_tickers, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                print("❌ Nenhum dado encontrado no Yahoo Finance")
                return pd.DataFrame()
            
            # Se houver apenas uma ação, ajusta o formato
            if len(yahoo_tickers) == 1:
                prices = pd.DataFrame(data['Adj Close'])
                prices.columns = [tickers[0]]
            else:
                prices = data['Adj Close']
                # Remove .SA dos nomes das colunas
                prices.columns = [col.replace('.SA', '') for col in prices.columns]
            
            # Remove colunas com todos os valores NaN
            prices = prices.dropna(axis=1, how='all')
            
            print(f"✅ Dados obtidos para {len(prices.columns)} ações")
            return prices
            
        except Exception as e:
            print(f"❌ Erro ao buscar dados: {e}")
            return pd.DataFrame()
    
    def calculate_returns(self, prices_df):
        """
        Calcula retornos das ações
        
        Args:
            prices_df (pd.DataFrame): DataFrame com preços históricos
            
        Returns:
            dict: Dicionário com retornos calculados
        """
        if prices_df.empty:
            return {}
        
        # Remove dados faltantes
        prices_clean = prices_df.dropna()
        
        if prices_clean.empty:
            return {}
        
        # Calcula retornos
        returns = {}
        
        for ticker in prices_clean.columns:
            first_price = prices_clean[ticker].iloc[0]
            last_price = prices_clean[ticker].iloc[-1]
            
            if pd.notna(first_price) and pd.notna(last_price) and first_price > 0:
                total_return = (last_price / first_price) - 1
                returns[ticker] = total_return
        
        return returns
    
    def plot_performance(self, prices_df, top_stocks, save_path=None):
        """
        Cria gráfico detalhado de performance das ações
        
        Args:
            prices_df (pd.DataFrame): Preços históricos
            top_stocks (pd.DataFrame): DataFrame com as top ações
            save_path (str): Caminho para salvar o gráfico
        """
        if prices_df.empty:
            print("❌ Não há dados para plotar")
            return
        
        # Calcula performance normalizada (base 100)
        normalized_prices = prices_df.div(prices_df.iloc[0]) * 100
        
        # Configuração da figura com subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Cores distintas para cada ação
        colors = plt.cm.tab10(np.linspace(0, 1, len(normalized_prices.columns)))
        
        # GRÁFICO PRINCIPAL - Todas as ações juntas
        ax1 = plt.subplot(3, 1, 1)
        
        # Plot das ações individuais
        for i, ticker in enumerate(normalized_prices.columns):
            if ticker in top_stocks['ticker'].values:
                final_performance = normalized_prices[ticker].iloc[-1]
                plt.plot(normalized_prices.index, normalized_prices[ticker], 
                        linewidth=2.5, label=f'{ticker} ({final_performance-100:+.1f}%)', 
                        color=colors[i], alpha=0.8)
        
        # Calcular e plotar média do portfólio
        portfolio_avg = normalized_prices.mean(axis=1)
        plt.plot(portfolio_avg.index, portfolio_avg.values, 
                linewidth=4, label=f'Portfólio Médio ({portfolio_avg.iloc[-1]-100:+.1f}%)', 
                color='red', linestyle='-', alpha=0.9)
        
        # Linha de referência
        plt.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Base (0%)')
        
        plt.title('Evolução das Top 10 Ações da Magic Formula - 2024\n(Performance Normalizada)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Performance (Base 100)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # GRÁFICO 2 - Grid 2x5 com ações individuais
        for i, ticker in enumerate(normalized_prices.columns[:10]):  # Top 10
            ax = plt.subplot(3, 5, i+6)  # Posições 6-15
            
            if ticker in top_stocks['ticker'].values:
                # Dados da ação
                stock_data = normalized_prices[ticker]
                final_return = stock_data.iloc[-1] - 100
                
                # Plot
                plt.plot(stock_data.index, stock_data.values, 
                        linewidth=2, color=colors[i], alpha=0.8)
                plt.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
                
                # Configurações
                plt.title(f'{ticker}\n{final_return:+.1f}%', fontsize=11, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45, fontsize=8)
                plt.yticks(fontsize=8)
                
                # Destacar cor baseada na performance
                if final_return > 0:
                    ax.patch.set_facecolor('lightgreen')
                    ax.patch.set_alpha(0.1)
                else:
                    ax.patch.set_facecolor('lightcoral')
                    ax.patch.set_alpha(0.1)
        
        # GRÁFICO 3 - Ranking de performance
        ax3 = plt.subplot(3, 1, 3)
        
        # Calcular retornos finais
        final_returns = {}
        for ticker in normalized_prices.columns:
            if ticker in top_stocks['ticker'].values:
                final_returns[ticker] = normalized_prices[ticker].iloc[-1] - 100
        
        # Ordenar por performance
        sorted_returns = dict(sorted(final_returns.items(), key=lambda x: x[1], reverse=True))
        
        # Criar gráfico de barras
        tickers_sorted = list(sorted_returns.keys())
        returns_sorted = list(sorted_returns.values())
        
        bars = plt.bar(range(len(tickers_sorted)), returns_sorted, 
                      color=['green' if x > 0 else 'red' for x in returns_sorted],
                      alpha=0.7, edgecolor='black', linewidth=1)
        
        # Adicionar valores nas barras
        for i, (ticker, ret) in enumerate(sorted_returns.items()):
            plt.text(i, ret + (1 if ret > 0 else -1), f'{ret:.1f}%', 
                    ha='center', va='bottom' if ret > 0 else 'top', fontweight='bold')
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        plt.title('Ranking de Performance - Top 10 Ações Magic Formula 2024', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Ações (ordenadas por performance)', fontsize=12)
        plt.ylabel('Retorno (%)', fontsize=12)
        plt.xticks(range(len(tickers_sorted)), tickers_sorted, rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar gráfico se especificado
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"💾 Gráfico detalhado salvo em: {save_path}")
        
        plt.show()
        
        # Imprimir resumo da performance
        print(f"\n📊 RESUMO DA PERFORMANCE INDIVIDUAL:")
        print("="*60)
        for i, (ticker, ret) in enumerate(sorted_returns.items(), 1):
            status = "🟢" if ret > 0 else "🔴"
            print(f"{i:2d}. {status} {ticker:6s}: {ret:+7.1f}%")
        
        # Estatísticas do portfólio
        avg_return = np.mean(list(final_returns.values()))
        median_return = np.median(list(final_returns.values()))
        std_return = np.std(list(final_returns.values()))
        
        print(f"\n📈 ESTATÍSTICAS DO PORTFÓLIO:")
        print(f"   • Retorno Médio: {avg_return:+.1f}%")
        print(f"   • Retorno Mediano: {median_return:+.1f}%")
        print(f"   • Desvio Padrão: {std_return:.1f}%")
        print(f"   • Melhor Ação: {max(final_returns.values()):+.1f}%")
        print(f"   • Pior Ação: {min(final_returns.values()):+.1f}%")
        print(f"   • Ações Positivas: {sum(1 for x in final_returns.values() if x > 0)}/10")
    
    def plot_individual_evolution(self, prices_df, top_stocks, save_path=None):
        """
        Cria gráficos individuais detalhados para cada ação
        
        Args:
            prices_df (pd.DataFrame): Preços históricos
            top_stocks (pd.DataFrame): DataFrame com as top ações
            save_path (str): Caminho base para salvar os gráficos
        """
        if prices_df.empty:
            print("❌ Não há dados para plotar")
            return
        
        print(f"\n📊 CRIANDO GRÁFICOS INDIVIDUAIS...")
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8-darkgrid')
        
        for ticker in prices_df.columns[:10]:  # Top 10 ações
            if ticker not in top_stocks['ticker'].values:
                continue
                
            # Dados da ação
            stock_data = top_stocks[top_stocks['ticker'] == ticker].iloc[0]
            price_series = prices_df[ticker].dropna()
            
            if len(price_series) < 2:
                continue
            
            # Calcular métricas
            initial_price = price_series.iloc[0]
            final_price = price_series.iloc[-1]
            total_return = (final_price / initial_price - 1) * 100
            
            # Volatilidade
            daily_returns = price_series.pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252) * 100
            
            # Drawdown
            running_max = price_series.expanding().max()
            drawdown = (price_series - running_max) / running_max * 100
            max_drawdown = drawdown.min()
            
            # Criar figura
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{ticker} - Análise Detalhada da Evolução (2024)', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # Gráfico 1: Preço Absoluto
            ax1.plot(price_series.index, price_series.values, linewidth=2, color='blue')
            ax1.set_title(f'Evolução do Preço\nR$ {initial_price:.2f} → R$ {final_price:.2f} ({total_return:+.1f}%)')
            ax1.set_ylabel('Preço (R$)')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Destacar início e fim
            ax1.scatter([price_series.index[0], price_series.index[-1]], 
                       [initial_price, final_price], 
                       color=['green', 'red' if total_return < 0 else 'darkgreen'], 
                       s=100, zorder=5)
            
            # Gráfico 2: Performance Normalizada
            normalized = price_series / initial_price * 100
            ax2.plot(normalized.index, normalized.values, linewidth=2, color='orange')
            ax2.axhline(y=100, color='black', linestyle='--', alpha=0.5)
            ax2.set_title(f'Performance Normalizada (Base 100)\nRetorno: {total_return:+.1f}%')
            ax2.set_ylabel('Performance (Base 100)')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # Colorir área baseada na performance
            ax2.fill_between(normalized.index, 100, normalized.values, 
                           alpha=0.3, color='green' if total_return > 0 else 'red')
            
            # Gráfico 3: Drawdown
            ax3.fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.3)
            ax3.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
            ax3.set_title(f'Drawdown Máximo: {max_drawdown:.1f}%')
            ax3.set_ylabel('Drawdown (%)')
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
            
            # Gráfico 4: Informações Fundamentais
            ax4.axis('off')
            
            # Criar tabela de informações
            info_text = f"""
INFORMAÇÕES FUNDAMENTAIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Magic Formula:
   • Posição no Ranking: {stock_data['posicao']}º
   • Earning Yield: {stock_data['earning_yield']:.1%}
   • ROIC: {stock_data['roic']:.1%}

💰 Preços:
   • Preço Inicial: R$ {initial_price:.2f}
   • Preço Final: R$ {final_price:.2f}
   • Retorno Total: {total_return:+.1f}%

📈 Métricas de Risco:
   • Volatilidade Anual: {volatility:.1f}%
   • Drawdown Máximo: {max_drawdown:.1f}%

🏢 Empresa:
   • Setor: {stock_data['setor'].title()}
   • P/L: {stock_data.get('p_l', 'N/A')}
   • P/VP: {stock_data.get('p_vp', 'N/A')}
            """
            
            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=1", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            
            # Salvar gráfico individual
            if save_path:
                individual_path = save_path.replace('.png', f'_{ticker}.png')
                plt.savefig(individual_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"   ✅ {ticker}: {individual_path}")
            
            plt.show()
        
        print(f"✅ Gráficos individuais criados com sucesso!")
    
    def create_evolution_heatmap(self, prices_df, top_stocks, save_path=None):
        """
        Cria heatmap da evolução mensal das ações
        
        Args:
            prices_df (pd.DataFrame): Preços históricos
            top_stocks (pd.DataFrame): DataFrame com as top ações
            save_path (str): Caminho para salvar o gráfico
        """
        if prices_df.empty:
            print("❌ Não há dados para plotar heatmap")
            return
        
        # Calcular retornos mensais
        monthly_returns = prices_df.resample('M').last().pct_change().dropna()
        
        if monthly_returns.empty:
            print("❌ Dados insuficientes para heatmap mensal")
            return
        
        # Converter para porcentagem
        monthly_returns_pct = monthly_returns * 100
        
        # Adicionar nomes dos meses
        month_names = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                      'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        
        monthly_returns_pct.index = [month_names[date.month-1] for date in monthly_returns_pct.index]
        
        # Filtrar apenas as ações top 10
        available_tickers = [ticker for ticker in top_stocks['ticker'].head(10) 
                           if ticker in monthly_returns_pct.columns]
        
        if not available_tickers:
            print("❌ Nenhuma ação disponível para heatmap")
            return
        
        monthly_data = monthly_returns_pct[available_tickers]
        
        # Criar heatmap
        plt.figure(figsize=(14, 8))
        
        # Configurar cores (verde para positivo, vermelho para negativo)
        sns.heatmap(monthly_data.T, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Retorno Mensal (%)'}, 
                   linewidths=0.5, linecolor='white')
        
        plt.title('Heatmap de Retornos Mensais - Top 10 Magic Formula 2024\n(% de retorno por mês)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Mês', fontsize=12)
        plt.ylabel('Ações', fontsize=12)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        # Adicionar linha separadora no meio do ano
        if len(monthly_data) > 6:
            plt.axvline(x=6, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            heatmap_path = save_path.replace('.png', '_heatmap.png')
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"💾 Heatmap salvo em: {heatmap_path}")
        
        plt.show()
        
        # Estatísticas do heatmap
        print(f"\n🔥 ANÁLISE DO HEATMAP:")
        print("="*50)
        
        # Melhor mês para cada ação
        best_months = monthly_data.idxmax()
        worst_months = monthly_data.idxmin()
        
        print(f"📈 MELHORES MESES POR AÇÃO:")
        for ticker in available_tickers:
            best_month = best_months[ticker]
            best_return = monthly_data.loc[best_month, ticker]
            print(f"   • {ticker}: {best_month} ({best_return:+.1f}%)")
        
        print(f"\n📉 PIORES MESES POR AÇÃO:")
        for ticker in available_tickers:
            worst_month = worst_months[ticker]
            worst_return = monthly_data.loc[worst_month, ticker]
            print(f"   • {ticker}: {worst_month} ({worst_return:+.1f}%)")
        
        # Melhor e pior mês geral
        monthly_avg = monthly_data.mean(axis=1)
        best_month_overall = monthly_avg.idxmax()
        worst_month_overall = monthly_avg.idxmin()
        
        print(f"\n🏆 DESEMPENHO MENSAL GERAL:")
        print(f"   • Melhor Mês: {best_month_overall} ({monthly_avg[best_month_overall]:+.1f}%)")
        print(f"   • Pior Mês: {worst_month_overall} ({monthly_avg[worst_month_overall]:+.1f}%)")
    def plot_magic_formula_analysis_from_api(self, top_stocks, save_path=None):
        """
        Cria gráfico de análise usando APENAS os dados da API do Laboratório de Finanças
        Não depende de dados externos (Yahoo Finance)
        
        Args:
            top_stocks (pd.DataFrame): DataFrame com as top ações da Magic Formula
            save_path (str): Caminho para salvar o gráfico (opcional)
        """
        if top_stocks.empty:
            print("❌ Não há dados para plotar")
            return
        
        print(f"\n🎨 Criando análise Magic Formula com dados da API...")
        
        # Criar figura com múltiplos subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📊 ANÁLISE MAGIC FORMULA - TOP 10 AÇÕES SELECIONADAS\n(Baseado em dados do Laboratório de Finanças)', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # 1. Gráfico de Earning Yield vs ROIC (Principal da Magic Formula)
        ax1.scatter(top_stocks['earning_yield'] * 100, top_stocks['roic'] * 100, 
                   s=200, alpha=0.7, c=range(len(top_stocks)), cmap='viridis')
        
        # Adicionar labels dos tickers
        for i, row in top_stocks.iterrows():
            ax1.annotate(f"{row['posicao']}º\n{row['ticker']}", 
                        (row['earning_yield'] * 100, row['roic'] * 100),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax1.set_xlabel('Earning Yield (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('ROIC (%)', fontsize=12, fontweight='bold')
        ax1.set_title('🎯 Magic Formula: Earning Yield vs ROIC', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#f8f9fa')
        
        # 2. Ranking das Ações
        colors = ['gold' if i < 3 else 'lightgreen' if i < 6 else 'lightcoral' for i in range(len(top_stocks))]
        bars = ax2.barh(range(len(top_stocks)), top_stocks['magic_formula_rank'], color=colors, alpha=0.8)
        
        ax2.set_yticks(range(len(top_stocks)))
        ax2.set_yticklabels([f"{row['posicao']}º {row['ticker']}" for _, row in top_stocks.iterrows()])
        ax2.set_xlabel('Score Magic Formula (menor = melhor)', fontsize=12, fontweight='bold')
        ax2.set_title('🏆 Ranking Magic Formula', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.invert_yaxis()
        
        # Adicionar valores nas barras
        for i, (bar, score) in enumerate(zip(bars, top_stocks['magic_formula_rank'])):
            ax2.text(score + max(top_stocks['magic_formula_rank']) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.1f}', ha='left', va='center', fontweight='bold')
        
        # 3. Distribuição por Setor
        sector_counts = top_stocks['setor'].value_counts()
        wedges, texts, autotexts = ax3.pie(sector_counts.values, labels=sector_counts.index, 
                                          autopct='%1.0f%%', startangle=90)
        ax3.set_title('🏭 Distribuição por Setor', fontweight='bold')
        
        # Melhorar aparência do pie chart
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # 4. Múltiplos de Avaliação
        x_pos = range(len(top_stocks))
        
        # P/L (se disponível)
        if 'p_l' in top_stocks.columns:
            p_l_values = pd.to_numeric(top_stocks['p_l'], errors='coerce')
            valid_pl = p_l_values.dropna()
            if not valid_pl.empty and valid_pl.max() < 100:  # Filtrar P/L extremos
                ax4.bar([i-0.2 for i in range(len(valid_pl))], valid_pl, 
                       width=0.4, alpha=0.7, label='P/L', color='skyblue')
        
        # P/VP (se disponível)
        if 'p_vp' in top_stocks.columns:
            p_vp_values = pd.to_numeric(top_stocks['p_vp'], errors='coerce')
            valid_pvp = p_vp_values.dropna()
            if not valid_pvp.empty and valid_pvp.max() < 10:  # Filtrar P/VP extremos
                ax4.bar([i+0.2 for i in range(len(valid_pvp))], valid_pvp, 
                       width=0.4, alpha=0.7, label='P/VP', color='lightcoral')
        
        ax4.set_xlabel('Ações (por ranking)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Múltiplos', fontsize=12, fontweight='bold')
        ax4.set_title('📈 Múltiplos de Avaliação', fontweight='bold')
        ax4.set_xticks(range(len(top_stocks)))
        ax4.set_xticklabels([row['ticker'] for _, row in top_stocks.iterrows()], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Salvar gráfico se especificado
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"💾 Gráfico salvo em: {save_path}")
        
        plt.show()
        
        # Imprimir análise detalhada dos dados da API
        self._print_api_data_analysis(top_stocks)
    
    def _print_api_data_analysis(self, top_stocks):
        """
        Imprime análise detalhada usando apenas dados da API
        
        Args:
            top_stocks (pd.DataFrame): DataFrame com as top ações
        """
        print(f"\n📊 ANÁLISE DETALHADA DOS DADOS DA API:")
        print("=" * 70)
        
        print(f"\n🏆 TOP 10 AÇÕES SELECIONADAS PELA MAGIC FORMULA:")
        print("-" * 70)
        
        for _, row in top_stocks.iterrows():
            # Dados básicos
            ticker = row['ticker']
            posicao = row['posicao']
            setor = row['setor']
            preco = row['preco']
            
            # Métricas Magic Formula
            earning_yield = row['earning_yield']
            roic = row['roic']
            rank_mf = row['magic_formula_rank']
            
            # Múltiplos (se disponíveis)
            p_l = row.get('p_l', 'N/A')
            p_vp = row.get('p_vp', 'N/A')
            ev_ebitda = row.get('ev_ebitda', 'N/A')
            
            print(f"\n{posicao:2d}º {ticker:6s} | {setor.title()[:15]:15s} | R$ {preco:7.2f}")
            print(f"    🎯 Magic Formula: EY={earning_yield:6.1%} | ROIC={roic:6.1%} | Score={rank_mf:5.1f}")
            
            # Mostrar múltiplos se disponíveis
            multiples = []
            if pd.notna(p_l) and p_l != 'N/A':
                multiples.append(f"P/L={p_l}")
            if pd.notna(p_vp) and p_vp != 'N/A':
                multiples.append(f"P/VP={p_vp}")
            if pd.notna(ev_ebitda) and ev_ebitda != 'N/A':
                multiples.append(f"EV/EBITDA={ev_ebitda}")
            
            if multiples:
                print(f"    📈 Múltiplos: {' | '.join(multiples)}")
        
        # Estatísticas gerais
        print(f"\n📊 ESTATÍSTICAS GERAIS:")
        print("-" * 40)
        
        avg_earning_yield = top_stocks['earning_yield'].mean()
        avg_roic = top_stocks['roic'].mean()
        avg_price = top_stocks['preco'].mean()
        
        print(f"   • Earning Yield Médio: {avg_earning_yield:6.1%}")
        print(f"   • ROIC Médio: {avg_roic:6.1%}")
        print(f"   • Preço Médio: R$ {avg_price:7.2f}")
        
        # Análise setorial
        sector_dist = top_stocks['setor'].value_counts()
        print(f"\n🏭 DISTRIBUIÇÃO SETORIAL:")
        print("-" * 30)
        for setor, count in sector_dist.items():
            pct = (count / len(top_stocks)) * 100
            print(f"   • {setor.title():20s}: {count:2d} ações ({pct:4.0f}%)")
        
        # Faixas de avaliação
        print(f"\n💰 FAIXAS DE PREÇO:")
        print("-" * 25)
        
        low_price = top_stocks[top_stocks['preco'] < 10]
        mid_price = top_stocks[(top_stocks['preco'] >= 10) & (top_stocks['preco'] < 50)]
        high_price = top_stocks[top_stocks['preco'] >= 50]
        
        print(f"   • Baixo (< R$ 10): {len(low_price)} ações")
        print(f"   • Médio (R$ 10-50): {len(mid_price)} ações") 
        print(f"   • Alto (> R$ 50): {len(high_price)} ações")
        
        # Melhores métricas
        best_ey = top_stocks.loc[top_stocks['earning_yield'].idxmax()]
        best_roic = top_stocks.loc[top_stocks['roic'].idxmax()]
        
        print(f"\n🥇 DESTAQUES:")
        print("-" * 20)
        print(f"   • Melhor Earning Yield: {best_ey['ticker']} ({best_ey['earning_yield']:.1%})")
        print(f"   • Melhor ROIC: {best_roic['ticker']} ({best_roic['roic']:.1%})")
        
        print(f"\n💡 INTERPRETAÇÃO:")
        print("-" * 20)
        print(f"   • Earning Yield > 10%: Empresas com bom retorno")
        print(f"   • ROIC > 15%: Empresas muito eficientes")
        print(f"   • Score MF baixo: Melhor combinação EY + ROIC")
        
        print(f"\n⚠️ NOTA: Análise baseada exclusivamente em dados do Laboratório de Finanças")
        print(f"         Período de referência: {top_stocks.iloc[0].get('data_base', 'N/A')}")
    
    def create_fundamental_evolution_chart(self, df_historical_api, save_path=None):
        """
        Cria gráfico de evolução usando dados históricos da própria API
        (se disponível múltiplas datas)
        
        Args:
            df_historical_api (pd.DataFrame): Dados históricos da API
            save_path (str): Caminho para salvar
        """
        if df_historical_api.empty:
            print("❌ Não há dados históricos da API")
            return
        
        print(f"\n📈 Criando evolução com dados da API...")
        
        # Agrupar por ticker e data
        if 'data_base' in df_historical_api.columns:
            # Organizar dados por data
            dates = sorted(df_historical_api['data_base'].unique())
            tickers = df_historical_api['ticker'].unique()[:10]  # Top 10
            
            plt.figure(figsize=(15, 10))
            
            # Para cada ticker, plotar evolução dos múltiplos
            for ticker in tickers:
                ticker_data = df_historical_api[df_historical_api['ticker'] == ticker]
                ticker_data = ticker_data.sort_values('data_base')
                
                if len(ticker_data) > 1:
                    # Usar P/L como proxy de evolução
                    if 'p_l' in ticker_data.columns:
                        p_l_values = pd.to_numeric(ticker_data['p_l'], errors='coerce')
                        valid_data = ticker_data[p_l_values.notna() & (p_l_values < 50)]
                        
                        if len(valid_data) > 1:
                            plt.plot(valid_data['data_base'], valid_data['p_l'], 
                                   marker='o', linewidth=2, label=ticker, alpha=0.8)
            
            plt.title('Evolução dos Múltiplos P/L - Top Ações Magic Formula\n(Dados da API do Laboratório de Finanças)', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Data')
            plt.ylabel('P/L')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 Gráfico de evolução salvo em: {save_path}")
            
            plt.show()
        else:
            print("❌ Dados da API não contêm múltiplas datas para evolução")
        """
        Cria gráfico básico de evolução das ações selecionadas pela Magic Formula
        
        Args:
            prices_df (pd.DataFrame): Preços históricos das ações
            top_stocks (pd.DataFrame): DataFrame com as top ações selecionadas
            save_path (str): Caminho para salvar o gráfico (opcional)
        """
        if prices_df.empty:
            print("❌ Não há dados para plotar")
            return
        
        print(f"\n🎨 Criando gráfico de evolução das ações...")
        
        # Calcular performance normalizada (base 100)
        normalized_prices = prices_df.div(prices_df.iloc[0]) * 100
        
        # Configurar cores para cada ação
        colors = plt.cm.tab10(np.linspace(0, 1, len(normalized_prices.columns)))
        
        # Criar figura
        plt.figure(figsize=(16, 10))
        
        # Plot das ações individuais
        for i, ticker in enumerate(normalized_prices.columns):
            if ticker in top_stocks['ticker'].values:
                final_performance = normalized_prices[ticker].iloc[-1]
                return_pct = final_performance - 100
                
                plt.plot(normalized_prices.index, normalized_prices[ticker], 
                        linewidth=2.5, 
                        label=f'{ticker} ({return_pct:+.1f}%)', 
                        color=colors[i], 
                        alpha=0.8)
        
        # Calcular e plotar média do portfólio Magic Formula
        portfolio_avg = normalized_prices.mean(axis=1)
        portfolio_return = portfolio_avg.iloc[-1] - 100
        
        plt.plot(portfolio_avg.index, portfolio_avg.values, 
                linewidth=4, 
                label=f'Portfólio Magic Formula ({portfolio_return:+.1f}%)', 
                color='red', 
                linestyle='-', 
                alpha=1.0)
        
        # Linha de referência (0% de retorno)
        plt.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Base (0%)')
        
        # Configurações do gráfico
        plt.title('Evolução das Top 10 Ações Magic Formula - 2024\n(Performance Normalizada - Base 100)', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Data', fontsize=14)
        plt.ylabel('Performance (Base 100)', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Melhorar aparência
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().set_facecolor('#f8f9fa')
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar gráfico se especificado
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"💾 Gráfico salvo em: {save_path}")
        
        # Mostrar gráfico
        plt.show()
        
        # Imprimir estatísticas de performance
        self._print_evolution_statistics(normalized_prices, top_stocks)
    
    def _print_evolution_statistics(self, normalized_prices, top_stocks):
        """
        Imprime estatísticas da evolução das ações
        
        Args:
            normalized_prices (pd.DataFrame): Preços normalizados
            top_stocks (pd.DataFrame): DataFrame com as top ações
        """
        print(f"\n📊 ESTATÍSTICAS DE PERFORMANCE:")
        print("=" * 60)
        
        # Calcular retornos finais
        final_returns = {}
        for ticker in normalized_prices.columns:
            if ticker in top_stocks['ticker'].values:
                final_returns[ticker] = normalized_prices[ticker].iloc[-1] - 100
        
        # Ordenar por performance
        sorted_returns = dict(sorted(final_returns.items(), key=lambda x: x[1], reverse=True))
        
        print(f"\n🏆 RANKING DE PERFORMANCE:")
        for i, (ticker, ret) in enumerate(sorted_returns.items(), 1):
            status = "🟢" if ret > 0 else "🔴"
            # Buscar informações da Magic Formula
            stock_info = top_stocks[top_stocks['ticker'] == ticker]
            if not stock_info.empty:
                position = stock_info.iloc[0]['posicao']
                earning_yield = stock_info.iloc[0]['earning_yield']
                roic = stock_info.iloc[0]['roic']
                print(f"{i:2d}. {status} {ticker:6s}: {ret:+7.1f}% | "
                      f"Pos.MF: {position:2d}º | EY: {earning_yield:5.1%} | ROIC: {roic:5.1%}")
            else:
                print(f"{i:2d}. {status} {ticker:6s}: {ret:+7.1f}%")
        
        # Estatísticas gerais do portfólio
        returns_list = list(final_returns.values())
        avg_return = np.mean(returns_list)
        median_return = np.median(returns_list)
        std_return = np.std(returns_list)
        positive_count = sum(1 for x in returns_list if x > 0)
        
        print(f"\n📈 RESUMO DO PORTFÓLIO MAGIC FORMULA:")
        print("-" * 50)
        print(f"   • Retorno Médio: {avg_return:+6.1f}%")
        print(f"   • Retorno Mediano: {median_return:+6.1f}%")
        print(f"   • Desvio Padrão: {std_return:6.1f}%")
        print(f"   • Ações Positivas: {positive_count:2d}/{len(returns_list):2d} ({positive_count/len(returns_list)*100:.0f}%)")
        print(f"   • Taxa de Sucesso: {positive_count/len(returns_list)*100:5.0f}%")
        print(f"   • Melhor Ação: {max(final_returns.values()):+6.1f}% ({max(final_returns, key=final_returns.get)})")
        print(f"   • Pior Ação: {min(final_returns.values()):+6.1f}% ({min(final_returns, key=final_returns.get)})")
        
        # Estatísticas da Magic Formula
        if not top_stocks.empty:
            avg_earning_yield = top_stocks['earning_yield'].mean()
            avg_roic = top_stocks['roic'].mean()
            
            print(f"\n🧮 MÉTRICAS MAGIC FORMULA (Médias):")
            print("-" * 40)
            print(f"   • Earning Yield Médio: {avg_earning_yield:6.1%}")
            print(f"   • ROIC Médio: {avg_roic:6.1%}")
        
        # Comparação com benchmarks (estimativa)
        print(f"\n📊 COMPARAÇÃO ESTIMADA:")
        print("-" * 30)
        print(f"   • Ibovespa 2024: ~-10.4%")
        print(f"   • CDI 2024: ~+11.2%")
        print(f"   • Magic Formula: {avg_return:+.1f}%")
        
        if avg_return > 11.2:
            print(f"   ✅ Estratégia superou CDI em {avg_return-11.2:.1f} p.p.")
        else:
            print(f"   ❌ Estratégia ficou {11.2-avg_return:.1f} p.p. abaixo do CDI")
        
        print(f"\n⚠️  Nota: Dados baseados em simulação para fins educacionais")
    
    def generate_report(self, top_stocks, returns_data):
        """
        Gera relatório final da análise
        
        Args:
            top_stocks (pd.DataFrame): Top ações selecionadas
            returns_data (dict): Dados de retorno das ações
        """
        print("\n" + "="*80)
        print("📊 RELATÓRIO FINAL - MAGIC FORMULA 2024")
        print("="*80)
        
        # Estatísticas gerais
        valid_returns = [ret for ret in returns_data.values() if not pd.isna(ret)]
        
        if valid_returns:
            avg_return = np.mean(valid_returns)
            median_return = np.median(valid_returns)
            best_return = max(valid_returns)
            worst_return = min(valid_returns)
            
            print(f"\n📈 PERFORMANCE GERAL (Jan-Dez 2024):")
            print(f"   • Retorno Médio: {avg_return:.1%}")
            print(f"   • Retorno Mediano: {median_return:.1%}")
            print(f"   • Melhor Retorno: {best_return:.1%}")
            print(f"   • Pior Retorno: {worst_return:.1%}")
        
        # Detalhamento por ação
        print(f"\n📋 DETALHAMENTO POR AÇÃO:")
        print("-" * 80)
        
        for i, row in top_stocks.iterrows():
            ticker = row['ticker']
            return_pct = returns_data.get(ticker, np.nan)
            
            return_str = f"{return_pct:.1%}" if not pd.isna(return_pct) else "N/A"
            
            print(f"{row['posicao']:2d}. {ticker:6s} | "
                  f"Retorno 2024: {return_str:>8s} | "
                  f"EY: {row['earning_yield']:6.1%} | "
                  f"ROIC: {row['roic']:6.1%}")
        
        print("\n" + "="*80)

def main():
    """Função principal para executar a análise completa"""
    
    # Configurações
    TOKEN = "dkadjkajljksajklajlknvn847824jk"  # Substitua pelo seu token real
    DATA_ANALISE = "2024-01-02"
    DATA_INICIO_PERFORMANCE = "2024-01-02"
    DATA_FIM_PERFORMANCE = "2024-12-30"
    TOP_N = 10
    
    print("🚀 INICIANDO ANÁLISE DA MAGIC FORMULA")
    print("="*50)
    
    # Inicializar analisador
    analyzer = MagicFormulaAnalyzer(TOKEN)
    
    # 1. Buscar dados da API
    print(f"\n1️⃣ Buscando dados para {DATA_ANALISE}...")
    df_raw = analyzer.get_data_from_api(DATA_ANALISE)
    
    if df_raw.empty:
        print("❌ Erro: Não foi possível obter dados da API")
        return
    
    # 2. Aplicar filtros
    print("\n2️⃣ Aplicando filtros da Magic Formula...")
    df_filtered = analyzer.apply_magic_formula_filters(df_raw)
    
    if df_filtered.empty:
        print("❌ Erro: Nenhuma ação passou nos filtros")
        return
    
    # 3. Calcular Magic Formula
    print("\n3️⃣ Calculando rankings da Magic Formula...")
    df_ranked = analyzer.calculate_magic_formula(df_filtered)
    
    # 4. Selecionar top ações
    print(f"\n4️⃣ Selecionando top {TOP_N} ações...")
    top_stocks = analyzer.get_top_stocks(df_ranked, TOP_N)
    
    # 5. Buscar performance histórica
    print(f"\n5️⃣ Analisando performance histórica...")
    tickers = top_stocks['ticker'].tolist()
    prices_df = analyzer.get_stock_performance(tickers, DATA_INICIO_PERFORMANCE, DATA_FIM_PERFORMANCE)
    
    # 6. Calcular retornos
    returns_data = analyzer.calculate_returns(prices_df)
    
    # 7. Gerar gráficos
    print(f"\n6️⃣ Gerando visualizações detalhadas...")
    
    # Gráfico básico de evolução das ações (NOVO MÉTODO)
   
    
    # Gráfico principal com todas as ações
    analyzer.plot_performance(prices_df, top_stocks, 'magic_formula_performance_2024.png')
    
    # Gráficos individuais para cada ação
    analyzer.plot_individual_evolution(prices_df, top_stocks, 'magic_formula_individual_2024.png')
    
    # Heatmap de retornos mensais
    analyzer.create_evolution_heatmap(prices_df, top_stocks, 'magic_formula_evolution_2024.png')
    
    # 8. Gerar relatório final
    print(f"\n7️⃣ Gerando relatório final...")
    analyzer.generate_report(top_stocks, returns_data)
    
    # 9. Salvar resultados em CSV
    print(f"\n8️⃣ Salvando resultados...")
    
    # Adicionar retornos ao DataFrame
    top_stocks['retorno_2024'] = top_stocks['ticker'].map(returns_data)
    
    # Salvar CSV
    output_file = 'magic_formula_top10_2024.csv'
    columns_to_save = ['posicao', 'ticker', 'setor', 'preco', 'earning_yield', 'roic', 
                      'magic_formula_rank', 'retorno_2024']
    
    top_stocks[columns_to_save].to_csv(output_file, index=False)
    print(f"💾 Resultados salvos em: {output_file}")
    
    print(f"\n✅ ANÁLISE CONCLUÍDA COM SUCESSO!")

if __name__ == "__main__":
    main()
 
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
        
    

    
  
    
    # Verificar argumentos da linha de comando
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_help()
    else:
        run_complete_analysis()