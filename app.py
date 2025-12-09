"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║              PORTFOLIO OPTIMIZER - Interfaccia Web Professionale              ║
║                        Powered by Streamlit & PyPortfolioOpt                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pypfopt import expected_returns, risk_models, EfficientFrontier, EfficientSemivariance, HRPOpt
from scipy.optimize import minimize, Bounds
import warnings
import io

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE PAGINA
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Portfolio Optimizer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Personalizzato
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATI E CONFIGURAZIONI
# ═══════════════════════════════════════════════════════════════════════════════

BENCHMARK_CANDIDATES = [
    {'ticker': 'SPY', 'name': 'S&P 500'},
    {'ticker': 'QQQ', 'name': 'NASDAQ 100'},
    {'ticker': 'VTI', 'name': 'Total US'},
    {'ticker': 'XLK', 'name': 'Tech SPDR'},
    {'ticker': 'SMH', 'name': 'Semiconductor'},
    {'ticker': 'VUG', 'name': 'Growth'},
]

DEFAULT_SECTOR_MAP = {
    'SXLK': 'Technology', 'XDWT': 'Technology', 'XLK': 'Technology', 'VGT': 'Technology', 'IYW': 'Technology',
    'CSNDX': 'NASDAQ', 'NQSE': 'NASDAQ', 'QQQ': 'NASDAQ', 'TQQQ': 'NASDAQ',
    'AIQ': 'AI', 'WTAI': 'AI', 'BOTZ': 'AI', 'ROBO': 'AI', 'IRBO': 'AI',
    'SMH': 'Semiconductor', 'SOXX': 'Semiconductor', 'FTXL': 'Semiconductor', 'HNSC': 'Semiconductor', 'PSI': 'Semiconductor',
    'WTEC': 'Cloud', 'SKYY': 'Cloud', 'CLOU': 'Cloud', 'XNGI': 'Cloud', 'WCLD': 'Cloud',
    'SIXG': 'EmergingTech', 'QTUM': 'EmergingTech', 'ARKQ': 'EmergingTech', 'ARKG': 'EmergingTech',
    'CTEK': 'CleanEnergy', 'ICLN': 'CleanEnergy', 'TAN': 'CleanEnergy', 'QCLN': 'CleanEnergy',
    'SEME': 'EmergingMarkets', 'EEM': 'EmergingMarkets', 'VWO': 'EmergingMarkets', 'IEMG': 'EmergingMarkets',
    'CIBR': 'Cybersecurity', 'HACK': 'Cybersecurity', 'BUG': 'Cybersecurity',
    'FINX': 'Fintech', 'ARKF': 'Fintech',
    'IBB': 'HealthTech', 'XBI': 'HealthTech',
}

DEFAULT_SECTOR_LIMITS = {
    'Technology': 0.35, 'NASDAQ': 0.30, 'AI': 0.25, 'Semiconductor': 0.30,
    'Cloud': 0.25, 'EmergingTech': 0.20, 'CleanEnergy': 0.20,
    'EmergingMarkets': 0.15, 'Cybersecurity': 0.20, 'Fintech': 0.15,
    'HealthTech': 0.20, 'Other': 0.25,
}

ETF_SUGGESTIONS = {
    'Technology': ['SXLK', 'XDWT', 'XLK', 'VGT', 'IYW'],
    'NASDAQ/Growth': ['CSNDX', 'NQSE', 'QQQ', 'VUG'],
    'AI & Robotics': ['AIQ', 'WTAI', 'BOTZ', 'ROBO', 'IRBO'],
    'Semiconductors': ['SMH', 'SOXX', 'FTXL', 'HNSC', 'PSI'],
    'Cloud Computing': ['WTEC', 'SKYY', 'CLOU', 'XNGI', 'WCLD'],
    'Emerging Tech': ['SIXG', 'QTUM', 'ARKQ', 'ARKG'],
    'Clean Energy': ['CTEK', 'ICLN', 'TAN', 'QCLN'],
    'Emerging Markets': ['SEME', 'EEM', 'VWO', 'IEMG'],
    'Cybersecurity': ['CIBR', 'HACK', 'BUG'],
    'Fintech': ['FINX', 'ARKF'],
    'HealthTech': ['IBB', 'XBI'],
}

# ═══════════════════════════════════════════════════════════════════════════════
# CLASSI DEL BACKEND (dal codice originale)
# ═══════════════════════════════════════════════════════════════════════════════

class BenchmarkAnalyzer:
    def __init__(self, start_date, end_date, risk_free_rate=0.02):
        self.start_date, self.end_date, self.risk_free_rate = start_date, end_date, risk_free_rate
        self.benchmark_prices, self.benchmark_returns, self.benchmark_metrics = {}, {}, {}
        self.best_benchmark = None

    def download_benchmark_data(self, ticker):
        if ticker in self.benchmark_prices:
            return True
        try:
            data = yf.Ticker(ticker).history(start=self.start_date, end=self.end_date, auto_adjust=True)
            if data.empty or len(data) < 50:
                return False
            prices = data['Close'].squeeze()
            prices.index = pd.to_datetime(prices.index.tz_localize(None) if prices.index.tz else prices.index).normalize()
            weekly = prices.resample('W').last().dropna()
            if isinstance(weekly, pd.DataFrame):
                weekly = weekly.squeeze()
            self.benchmark_prices[ticker] = weekly
            self.benchmark_returns[ticker] = weekly.pct_change().dropna() * 100
            return len(self.benchmark_returns[ticker]) >= 20
        except:
            return False

    def calculate_metrics(self, ticker):
        if not self.download_benchmark_data(ticker):
            return None
        ret, prices = self.benchmark_returns[ticker], self.benchmark_prices[ticker]
        mu = float(expected_returns.mean_historical_return(pd.DataFrame({ticker: prices}), frequency=52).iloc[0]) * 100
        rd = ret / 100
        vol = float(rd.std() * np.sqrt(52) * 100)
        excess = mu - self.risk_free_rate * 100
        sharpe = excess / vol if vol > 0 else 0
        ds = rd[rd < self.risk_free_rate/52]
        dd = float(np.sqrt(((ds - self.risk_free_rate/52)**2).mean()) * np.sqrt(52) * 100) if len(ds) > 0 else vol * 0.7
        sortino = excess / dd if dd > 0 else 0
        cum = (1 + rd).cumprod()
        mdd = float(abs(((cum - cum.expanding().max()) / cum.expanding().max()).min()) * 100)
        self.benchmark_metrics[ticker] = {
            'ticker': ticker, 'mean_return': mu, 'volatility': vol,
            'sharpe': sharpe, 'sortino': sortino, 'max_drawdown': mdd,
            'calmar': mu/mdd if mdd > 0 else 0
        }
        return self.benchmark_metrics[ticker]

    def find_best(self, port_returns, port_metrics):
        results = []
        for b in BENCHMARK_CANDIDATES:
            m = self.calculate_metrics(b['ticker'])
            if not m:
                continue
            br = self.benchmark_returns[b['ticker']].copy()
            pr = port_returns.copy()
            pr.index = pd.to_datetime(pr.index).normalize()
            br.index = pd.to_datetime(br.index).normalize()
            common = pr.index.intersection(br.index)
            if len(common) < 20:
                continue
            corr = float(pr.loc[common].corr(br.loc[common]))
            te = float((pr.loc[common] - br.loc[common]).std() * np.sqrt(52))
            cov = np.cov(pr.loc[common]/100, br.loc[common]/100)
            beta = cov[0,1]/cov[1,1] if cov[1,1] > 0 else 1.0
            score = corr * 0.5 + (1/(1+te/10)) * 0.35 + (1/(1+abs(m['volatility']-port_metrics['vol'])/10)) * 0.15
            results.append({**m, 'name': b['name'], 'correlation': corr, 'tracking_error': te, 'beta': beta, 'score': score})
        if not results:
            return None
        self.best_benchmark = max(results, key=lambda x: x['score'])
        return self.best_benchmark


class PortfolioOptimizer:
    def __init__(self, tickers, start_date='2020-01-01', end_date=None, min_weight=0.01,
                 risk_free_rate=0.02, max_concentration=0.25,
                 sector_map=None, sector_limits=None, target_volatility=None):
        self.tickers = [t.upper() for t in tickers]
        self.n_assets = len(tickers)
        self.start_date, self.end_date = start_date, end_date or datetime.today().strftime('%Y-%m-%d')
        self.min_weight = min_weight
        self.max_concentration = max_concentration
        self.risk_free_rate = risk_free_rate
        self.prices = self.returns = self.mu = self.S = None
        self.bench = BenchmarkAnalyzer(start_date, self.end_date, risk_free_rate)
        self.best_benchmark = None
        self.results = {}
        
        if sector_map is None:
            self.sector_map = DEFAULT_SECTOR_MAP.copy()
        else:
            self.sector_map = sector_map
        
        if sector_limits is False:
            self.sector_limits = None
            self.use_sector_constraints = False
        elif sector_limits is None:
            self.sector_limits = DEFAULT_SECTOR_LIMITS.copy()
            self.use_sector_constraints = True
        else:
            self.sector_limits = sector_limits
            self.use_sector_constraints = True
        
        self.target_volatility = target_volatility
        self.use_volatility_constraint = target_volatility is not None

    def download_data(self):
        all_data = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, t in enumerate(self.tickers, 1):
            status_text.text(f"📥 Download {t}... ({i}/{self.n_assets})")
            progress_bar.progress(i / self.n_assets)
            
            try:
                data = yf.Ticker(t).history(start=self.start_date, end=self.end_date, auto_adjust=True)
                if data.empty or len(data) < 50:
                    for sfx in ['.L', '.DE', '.MI']:
                        try:
                            data = yf.Ticker(t + sfx).history(start=self.start_date, end=self.end_date, auto_adjust=True)
                            if not data.empty and len(data) >= 50:
                                break
                        except:
                            continue
                if data.empty or len(data) < 50:
                    continue
                prices = data['Close'].squeeze()
                prices.index = pd.to_datetime(prices.index.tz_localize(None) if prices.index.tz else prices.index).normalize()
                all_data[t] = prices
            except Exception as e:
                st.warning(f"⚠️ Errore con {t}: {e}")
        
        progress_bar.empty()
        status_text.empty()
        
        self.tickers = list(all_data.keys())
        self.n_assets = len(self.tickers)
        if self.n_assets < 2:
            st.error("❌ Servono almeno 2 ETF validi!")
            return False
        
        df = pd.DataFrame(all_data).ffill(limit=5).bfill(limit=5).dropna()
        self.prices = df.resample('W').last().dropna()
        self.returns = self.prices.pct_change().dropna() * 100
        self.mu = expected_returns.mean_historical_return(self.prices, frequency=52)
        self.S = risk_models.sample_cov(self.prices, frequency=52)
        
        st.success(f"✅ {len(self.returns)} settimane di dati caricati per {self.n_assets} ETF")
        return True

    def _build_sector_mapper(self):
        return {ticker: self.sector_map.get(ticker, 'Other') for ticker in self.tickers}

    def _get_active_sectors(self):
        sector_mapper = self._build_sector_mapper()
        sectors = {}
        for ticker, sector in sector_mapper.items():
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(ticker)
        return sectors

    def _apply_sector_constraints(self, ef):
        if not self.use_sector_constraints:
            return ef
        sector_mapper = self._build_sector_mapper()
        sector_lower, sector_upper = {}, {}
        active_sectors = set(sector_mapper.values())
        for sector in active_sectors:
            sector_lower[sector] = 0
            sector_upper[sector] = self.sector_limits.get(sector, self.sector_limits.get('Other', 1.0))
        try:
            ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
        except Exception as e:
            st.warning(f"⚠️ Errore vincoli settoriali: {e}")
        return ef

    def stats(self, w):
        pr = (self.returns * w).sum(axis=1) / 100
        tot = (1 + pr).prod()
        n = len(pr) / 52
        ret = (tot ** (1/n) - 1) * 100 if n > 0 else 0
        vol = float(pr.std() * np.sqrt(52) * 100)
        excess = ret - self.risk_free_rate * 100
        sharpe = excess / vol if vol > 0 else 0
        ds = pr[pr < self.risk_free_rate/52]
        dd = float(np.sqrt(((ds - self.risk_free_rate/52)**2).mean()) * np.sqrt(52) * 100) if len(ds) > 0 else vol * 0.7
        sortino = excess / dd if dd > 0 else 0
        cum = (1 + pr).cumprod()
        mdd = float(abs(((cum - cum.expanding().max()) / cum.expanding().max()).min()) * 100)
        return {
            'ret': ret, 'vol': vol, 'sharpe': sharpe,
            'sortino': sortino, 'mdd': mdd,
            'calmar': ret/mdd if mdd > 0 else 0
        }

    def optimize_max_sharpe(self):
        ef = EfficientFrontier(self.mu, self.S, weight_bounds=(self.min_weight, self.max_concentration))
        if self.use_sector_constraints:
            ef = self._apply_sector_constraints(ef)
        if self.use_volatility_constraint:
            try:
                ef.efficient_risk(target_volatility=self.target_volatility)
            except:
                ef = EfficientFrontier(self.mu, self.S, weight_bounds=(self.min_weight, self.max_concentration))
                if self.use_sector_constraints:
                    ef = self._apply_sector_constraints(ef)
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        else:
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        w = np.array([ef.clean_weights().get(t, 0) for t in self.tickers])
        s = self.stats(w)
        self.results['sharpe'] = {'weights': w, **s}
        return w

    def optimize_max_sortino(self):
        es = EfficientSemivariance(self.mu, self.prices.pct_change().dropna(), frequency=52,
                                   weight_bounds=(self.min_weight, self.max_concentration))
        if self.use_sector_constraints:
            es = self._apply_sector_constraints(es)
        if self.use_volatility_constraint:
            try:
                target_semidev = self.target_volatility * 0.8
                es.efficient_risk(target_semideviation=target_semidev)
            except:
                es = EfficientSemivariance(self.mu, self.prices.pct_change().dropna(), frequency=52,
                                          weight_bounds=(self.min_weight, self.max_concentration))
                if self.use_sector_constraints:
                    es = self._apply_sector_constraints(es)
                es.max_quadratic_utility(risk_aversion=1)
        else:
            es.max_quadratic_utility(risk_aversion=1)
        w = np.array([es.clean_weights().get(t, 0) for t in self.tickers])
        s = self.stats(w)
        self.results['sortino'] = {'weights': w, **s}
        return w

    def optimize_risk_parity(self):
        cov = self.S.values * 10000
        cov_annual = self.S.values

        def obj(w):
            pv = np.sqrt(w.T @ cov @ w)
            if pv < 1e-10:
                return 1e10
            rc = w * (cov @ w) / pv
            return np.sum((rc - pv/self.n_assets)**2)

        iv = 1 / np.sqrt(np.diag(cov))
        init = iv / iv.sum()
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        if self.use_sector_constraints:
            sector_mapper = self._build_sector_mapper()
            active_sectors = self._get_active_sectors()
            for sector, tickers_in_sector in active_sectors.items():
                max_weight = self.sector_limits.get(sector, self.sector_limits.get('Other', 1.0))
                indices = [self.tickers.index(t) for t in tickers_in_sector if t in self.tickers]
                if indices:
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda w, idx=indices, mw=max_weight: mw - np.sum(w[idx])
                    })

        if self.use_volatility_constraint:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.target_volatility**2 - (w.T @ cov_annual @ w)
            })

        res = minimize(obj, init, method='SLSQP',
                      bounds=Bounds([self.min_weight]*self.n_assets, [self.max_concentration]*self.n_assets),
                      constraints=constraints)
        w = res.x
        s = self.stats(w)
        self.results['rp'] = {'weights': w, **s}
        return w

    def optimize_hrp(self):
        hrp = HRPOpt(self.prices.pct_change().dropna())
        hrp.optimize()
        w = np.array([hrp.clean_weights().get(t, 0) for t in self.tickers])

        if self.use_volatility_constraint:
            cov_annual = self.S.values
            port_vol = np.sqrt(w.T @ cov_annual @ w)
            if port_vol > self.target_volatility:
                scale_factor = self.target_volatility / port_vol
                w = w * scale_factor
                w = w / w.sum()

        s = self.stats(w)
        self.results['hrp'] = {'weights': w, **s}
        return w

    def find_benchmark(self):
        eq = np.array([1/self.n_assets] * self.n_assets)
        pr = (self.returns * eq).sum(axis=1)
        s = self.stats(eq)
        self.best_benchmark = self.bench.find_best(pr, {'vol': s['vol'], 'ret': s['ret']})

    def calc_bench_metrics(self):
        if not self.best_benchmark:
            return
        br = self.bench.benchmark_returns[self.best_benchmark['ticker']].copy()
        br.index = pd.to_datetime(br.index).normalize()

        for name, data in self.results.items():
            pr = (self.returns * data['weights']).sum(axis=1)
            pr.index = pd.to_datetime(pr.index).normalize()
            common = pr.index.intersection(br.index)
            if len(common) < 20:
                continue
            pa, ba = pr.loc[common], br.loc[common]
            cov = np.cov(pa/100, ba/100)
            beta = cov[0,1]/cov[1,1] if cov[1,1] > 0 else 1.0
            te = float((pa - ba).std() * np.sqrt(52))
            n = len(pa)/52
            pc = ((1+pa/100).prod()**(1/n)-1)*100
            bc = ((1+ba/100).prod()**(1/n)-1)*100
            alpha = pc - bc
            ir = alpha/te if te > 0 else 0
            excess_return = data['ret'] - self.risk_free_rate * 100
            treynor = excess_return / beta if beta != 0 else 0
            calmar = data['ret'] / data['mdd'] if data['mdd'] > 0 else 0
            
            self.results[name].update({
                'beta': beta, 'te': te, 'alpha': alpha,
                'ir': ir, 'treynor': treynor, 'calmar': calmar,
                'n_weeks': len(common)
            })

    def run_all_optimizations(self):
        with st.spinner("🔄 Ottimizzazione in corso..."):
            self.optimize_max_sharpe()
            self.optimize_max_sortino()
            self.optimize_risk_parity()
            self.optimize_hrp()
            self.find_benchmark()
            self.calc_bench_metrics()
        st.success("✅ Ottimizzazione completata!")


# ═══════════════════════════════════════════════════════════════════════════════
# FUNZIONI DI VISUALIZZAZIONE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_efficient_frontier(optimizer):
    st.subheader("📈 Frontiera Efficiente")
    pts = []
    for t in np.linspace(float(optimizer.mu.min()), float(optimizer.mu.max()), 40):
        try:
            ef = EfficientFrontier(optimizer.mu, optimizer.S, 
                                  weight_bounds=(optimizer.min_weight, optimizer.max_concentration))
            ef.efficient_return(t)
            w = np.array([ef.clean_weights().get(tk, 0) for tk in optimizer.tickers])
            s = optimizer.stats(w)
            pts.append((s['vol'], s['ret'], s['sharpe']))
        except:
            pass
    
    if not pts:
        st.warning("⚠️ Impossibile generare la frontiera efficiente")
        return
    
    v, r, sh = zip(*pts)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=v, y=r, mode='markers',
        marker=dict(size=8, color=sh, colorscale='Viridis', showscale=True,
                   colorbar=dict(title="Sharpe Ratio")),
        name='Frontiera Efficiente'
    ))
    
    colors = {'sharpe': 'red', 'sortino': 'purple', 'rp': 'orange', 'hrp': 'cyan'}
    symbols = {'sharpe': 'star', 'sortino': 'diamond', 'rp': 'square', 'hrp': 'cross'}
    
    for name, data in optimizer.results.items():
        fig.add_trace(go.Scatter(
            x=[data['vol']], y=[data['ret']],
            mode='markers', name=name.upper(),
            marker=dict(size=20, color=colors.get(name, 'gray'), 
                       symbol=symbols.get(name, 'circle'),
                       line=dict(width=2, color='black'))
        ))
    
    fig.update_layout(
        title='Frontiera Efficiente',
        xaxis_title='Volatilità Annualizzata (%)',
        yaxis_title='Rendimento Annualizzato (%)',
        height=600,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
_returns(optimizer):
    st.subheader("📊 Rendimenti Cumulativi")
    
    fig = go.Figure()
    norm = optimizer.prices / optimizer.prices.iloc[0] * 100
    colors = {'sharpe': '#2ecc71', 'sortino': '#9b59b6', 'rp': '#f39c12', 'hrp': '#00bcd4'}
    
    for name, data in optimizer.results.items():
        cumulative = (norm * data['weights']).sum(axis=1)
        fig.add_trace(go.Scatter(
            x=cumulative.index, y=cumulative.values,
            mode='lines', name=name.upper(),
            line=dict(width=3, color=colors.get(name, 'gray'))
        ))
    
    if optimizer.best_benchmark and optimizer.best_benchmark['ticker'] in optimizer.bench.benchmark_prices:
        bp = optimizer.bench.benchmark_prices[optimizer.best_benchmark['ticker']]
        common = norm.index.intersection(bp.index)
        if len(common) > 0:
            bench_cum = bp.loc[common]/bp.loc[common].iloc[0]*100
            fig.add_trace(go.Scatter(
                x=bench_cum.index, y=bench_cum.values,
                mode='lines', name=f"Benchmark: {optimizer.best_benchmark['ticker']}",
                line=dict(width=2, dash='dash', color='blue')
            ))
    
    fig.add_hline(y=100, line_dash="dot", line_color="gray", annotation_text="Base 100€")
    
    fig.update_layout(
        title='Evoluzione del Portafoglio (Base 100€)',
        xaxis_title='Data',
        yaxis_title='Valore (€)',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_allocations(optimizer):
    st.subheader("🎯 Allocazioni per Strategia")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors_map = {'sharpe': '#2ecc71', 'sortino': '#9b59b6', 'rp': '#f39c12', 'hrp': '#00bcd4'}
    
    for idx, (name, data) in enumerate(optimizer.results.items()):
        ax = axes[idx // 2, idx % 2]
        si = np.argsort(data['weights'])[::-1]
        tks = [optimizer.tickers[i] for i in si if data['weights'][i] > 0.005]
        wts = [data['weights'][i]*100 for i in si if data['weights'][i] > 0.005]
        
        ax.barh(tks, wts, color=colors_map.get(name, 'gray'))
        ax.set_xlabel('Peso (%)', fontsize=11)
        ax.set_title(f"{name.upper()}\n(Sharpe: {data['sharpe']:.2f})", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('ALLOCAZIONI PER STRATEGIA', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    st.pyplot(fig)


def plot_drawdown(optimizer):
    st.subheader("📉 Drawdown nel Tempo")
    
    fig = go.Figure()
    colors = {'sharpe': '#2ecc71', 'sortino': '#9b59b6', 'rp': '#f39c12', 'hrp': '#00bcd4'}
    
    for name, data in optimizer.results.items():
        pr = (optimizer.returns * data['weights']).sum(axis=1)/100
        cum = (1+pr).cumprod()
        dd = (cum - cum.expanding().max())/cum.expanding().max()*100
        
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            mode='lines', name=f"{name.upper()} (Max: {dd.min():.1f}%)",
            line=dict(width=2, color=colors.get(name, 'gray')),
            fill='tozeroy', fillcolor=colors.get(name, 'gray'),
            opacity=0.3
        ))
    
    fig.update_layout(
        title='Drawdown: Perdita dal Massimo Storico',
        xaxis_title='Data',
        yaxis_title='Drawdown (%)',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_metrics_table(optimizer):
    st.subheader("📊 Tabella Metriche Complete")
    
    data_rows = []
    for name, data in optimizer.results.items():
        row = {
            'Strategia': name.upper(),
            'Rendimento %': f"{data['ret']:.2f}",
            'Volatilità %': f"{data['vol']:.2f}",
            'Max Drawdown %': f"{data['mdd']:.2f}",
            'Sharpe': f"{data['sharpe']:.3f}",
            'Sortino': f"{data['sortino']:.3f}",
            'Calmar': f"{data.get('calmar', 0):.3f}",
        }
        
        if 'beta' in data:
            row.update({
                'Beta': f"{data['beta']:.3f}",
                'Alpha %': f"{data['alpha']:+.2f}",
                'Tracking Error %': f"{data['te']:.2f}",
                'Info Ratio': f"{data['ir']:.3f}",
                'Treynor': f"{data['treynor']:.3f}",
            })
        
        data_rows.append(row)
    
    if optimizer.best_benchmark:
        b = optimizer.best_benchmark
        bench_row = {
            'Strategia': f"BENCHMARK ({b['ticker']})",
            'Rendimento %': f"{b['mean_return']:.2f}",
            'Volatilità %': f"{b['volatility']:.2f}",
            'Max Drawdown %': f"{b['max_drawdown']:.2f}",
            'Sharpe': f"{b['sharpe']:.3f}",
            'Sortino': f"{b['sortino']:.3f}",
            'Calmar': f"{b['calmar']:.3f}",
        }
        if 'beta' in data_rows[0]:
            bench_row.update({
                'Beta': "1.000",
                'Alpha %': "+0.00",
                'Tracking Error %': "0.00",
                'Info Ratio': "0.000",
                'Treynor': f"{(b['mean_return'] - optimizer.risk_free_rate*100):.3f}",
            })
        data_rows.append(bench_row)
    
    df = pd.DataFrame(data_rows)
    
    st.dataframe(
        df.style.apply(lambda x: ['background-color: #d4edda' if 'SHARPE' in str(x['Strategia']) 
                                  else 'background-color: #f8d7da' if 'BENCHMARK' in str(x['Strategia'])
                                  else '' for i in x], axis=1),
        use_container_width=True,
        height=400
    )


def display_interpretation_guide():
    with st.expander("📖 Guida Interpretazione Metriche", expanded=False):
        st.markdown("""
        ### 📊 Metriche Assolute
        
        - **Rendimento %**: Guadagno medio annuo atteso
          - > 8%: Buono | > 12%: Eccellente
        
        - **Volatilità %**: Rischio totale (oscillazioni)
          - < 15%: Conservativo | > 25%: Aggressivo
        
        - **Max Drawdown %**: Peggior perdita dal picco
          - < 15%: Ottimo | > 30%: Alto rischio
        
        - **Sharpe Ratio**: Rendimento per unità di rischio totale
          - > 1.0: Buono | > 2.0: Eccellente
        
        - **Sortino Ratio**: Come Sharpe, ma penalizza solo il rischio negativo
          - > 1.5: Buono | > 3.0: Eccellente
        
        - **Calmar Ratio**: Rendimento diviso Max Drawdown
          - > 1.0: Buono | > 3.0: Eccellente
        
        ---
        
        ### 📈 Metriche Relative al Benchmark
        
        - **Beta**: Sensibilità al mercato
          - β = 1: Segue il mercato | β > 1: Più volatile | β < 1: Meno volatile
        
        - **Alpha %**: Extra-rendimento vs benchmark
          - > 0: Batte il mercato | < 0: Sottoperforma
        
        - **Tracking Error %**: Quanto devia dal benchmark
          - < 5%: Gestione passiva | > 10%: Gestione attiva
        
        - **Information Ratio**: Alpha per unità di rischio attivo
          - > 0.5: Buono | > 1.0: Eccellente
        
        - **Treynor Ratio**: Rendimento per unità di rischio sistematico
          - Più alto è meglio
        
        ---
        
        ### 🎯 Quale Strategia Scegliere?
        
        - **MAX SHARPE**: Massimo rapporto rendimento/rischio totale
        - **MAX SORTINO**: Minimizza perdite (ideale per avversi al rischio)
        - **RISK PARITY**: Diversificazione equilibrata
        - **HRP**: Approccio robusto e stabile
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERFACCIA PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown('<div class="main-header">📊 Portfolio Optimizer Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ottimizzazione Professionale con PyPortfolioOpt</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/_static/img/streamlit-mark-color.png", width=100)
        st.title("⚙️ Configurazione")
        
        st.markdown("---")
        st.subheader("📅 Periodo Analisi")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Data Inizio",
                value=datetime.now() - timedelta(days=3*365),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "Data Fine",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        st.markdown("---")
        st.subheader("📋 Selezione ETF")
        
        selection_mode = st.radio(
            "Modalità selezione:",
            ["Rapida (suggerimenti)", "Manuale (inserisci ticker)"]
        )
        
        tickers = []
        
        if selection_mode == "Rapida (suggerimenti)":
            st.info("💡 Seleziona le categorie di interesse")
            selected_etfs = []
            
            for category, etf_list in ETF_SUGGESTIONS.items():
                with st.expander(f"📁 {category}"):
                    for etf in etf_list:
                        if st.checkbox(f"{etf} - {DEFAULT_SECTOR_MAP.get(etf, 'Other')}", key=etf):
                            selected_etfs.append(etf)
            
            tickers = selected_etfs
        else:
            ticker_input = st.text_area(
                "Inserisci ticker (uno per riga o separati da virgola):",
                height=150,
                placeholder="SXLK\nQQQ\nSMH\n..."
            )
            
            if ticker_input:
                if ',' in ticker_input:
                    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
                else:
                    tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]
        
        st.markdown("---")
        st.subheader("🎚️ Parametri Portafoglio")
        
        min_weight = st.slider(
            "Peso Minimo per Asset (%)",
            min_value=0.0, max_value=10.0, value=1.0, step=0.5
        ) / 100
        
        max_concentration = st.slider(
            "Peso Massimo per Asset (%)",
            min_value=10.0, max_value=100.0, value=25.0, step=5.0
        ) / 100
        
        risk_free_rate = st.slider(
            "Tasso Risk-Free (%)",
            min_value=0.0, max_value=10.0, value=3.7, step=0.1
        ) / 100
        
        st.markdown("---")
        st.subheader("🔒 Vincoli Avanzati")
        
        use_sector_limits = st.checkbox("Abilita vincoli settoriali", value=True)
        
        sector_limits_config = None
        if use_sector_limits:
            st.info("📊 Limiti massimi per settore")
            sector_limits_config = {}
            
            active_sectors = set([DEFAULT_SECTOR_MAP.get(t, 'Other') for t in tickers])
            
            for sector in sorted(active_sectors):
                default_limit = DEFAULT_SECTOR_LIMITS.get(sector, 25.0)
                sector_limits_config[sector] = st.slider(
                    f"{sector} (%)",
                    min_value=0.0, max_value=100.0, 
                    value=default_limit*100, step=5.0,
                    key=f"sector_{sector}"
                ) / 100
        else:
            sector_limits_config = False
        
        use_vol_cap = st.checkbox("Abilita vincolo volatilità", value=False)
        target_volatility = None
        
        if use_vol_cap:
            target_volatility = st.slider(
                "Volatilità Massima (%)",
                min_value=5.0, max_value=50.0, value=22.0, step=1.0
            ) / 100
        
        st.markdown("---")
        
        run_optimization = st.button("🚀 AVVIA OTTIMIZZAZIONE", type="primary", use_container_width=True)
    
    # Main Content
    if not tickers:
        st.info("👈 Seleziona almeno 2 ETF dalla sidebar per iniziare")
        
        st.markdown("---")
        st.subheader("📚 Come Usare l'App")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 1️⃣ Seleziona ETF
            - Usa i suggerimenti rapidi
            - Oppure inserisci ticker manualmente
            - Minimo 2 ETF richiesti
            """)
        
        with col2:
            st.markdown("""
            ### 2️⃣ Configura Parametri
            - Imposta periodo di analisi
            - Definisci pesi min/max
            - Abilita vincoli opzionali
            """)
        
        with col3:
            st.markdown("""
            ### 3️⃣ Analizza Risultati
            - 4 strategie ottimizzate
            - Grafici interattivi
            - Metriche dettagliate
            """)
        
        st.markdown("---")
        st.subheader("🎯 Strategie Disponibili")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **MAX SHARPE**
            - Massimizza rendimento/rischio totale
            - Approccio classico Markowitz
            - Ideale per investitori bilanciati
            
            **RISK PARITY**
            - Equalizza contributi al rischio
            - Portafoglio diversificato
            - Riduce dipendenza da singoli asset
            """)
        
        with col2:
            st.markdown("""
            **MAX SORTINO**
            - Penalizza solo rischio negativo
            - Per investitori avversi alle perdite
            - Focus su protezione capitale
            
            **HRP (Hierarchical Risk Parity)**
            - Allocazione gerarchica robusta
            - Resistente a errori di stima
            - Ottime performance out-of-sample
            """)
        
        return
    
    if len(tickers) < 2:
        st.error("❌ Servono almeno 2 ETF per l'ottimizzazione!")
        return
    
    # Display selected ETFs
    st.success(f"✅ {len(tickers)} ETF selezionati: {', '.join(tickers)}")
    
    if run_optimization:
        try:
            # Initialize optimizer
            optimizer = PortfolioOptimizer(
                tickers=tickers,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                min_weight=min_weight,
                max_concentration=max_concentration,
                risk_free_rate=risk_free_rate,
                sector_limits=sector_limits_config,
                target_volatility=target_volatility
            )
            
            # Download data
            st.markdown("---")
            st.subheader("📥 Download Dati Storici")
            
            if not optimizer.download_data():
                st.error("❌ Errore nel download dei dati")
                return
            
            # Run optimizations
            st.markdown("---")
            optimizer.run_all_optimizations()
            
            # Display results in tabs
            st.markdown("---")
            
            tabs = st.tabs([
                "📊 Panoramica",
                "📈 Grafici",
                "🎯 Allocazioni",
                "📉 Rischio",
                "📋 Tabelle",
                "💾 Download"
            ])
            
            with tabs[0]:
                st.subheader("🎯 Risultati Principali")
                
                # Metrics cards
                cols = st.columns(4)
                
                for idx, (name, data) in enumerate(optimizer.results.items()):
                    with cols[idx]:
                        st.metric(
                            label=name.upper(),
                            value=f"{data['ret']:.2f}%",
                            delta=f"Sharpe: {data['sharpe']:.2f}"
                        )
                        st.caption(f"Vol: {data['vol']:.2f}% | MDD: {data['mdd']:.2f}%")
                
                st.markdown("---")
                
                # Benchmark info
                if optimizer.best_benchmark:
                    st.info(f"🎯 Benchmark Selezionato: **{optimizer.best_benchmark['ticker']}** ({optimizer.best_benchmark['name']})")
                
                display_interpretation_guide()
            
            with tabs[1]:
                plot_efficient_frontier(optimizer)
                st.markdown("---")
                plot_cumulative_returns(optimizer)
            
            with tabs[2]:
                plot_allocations(optimizer)
            
            with tabs[3]:
                plot_drawdown(optimizer)
            
            with tabs[4]:
                display_metrics_table(optimizer)
            
            with tabs[5]:
                st.subheader("💾 Scarica Risultati")
                
                # Prepare CSV
                rows = [{
                    'ETF': t,
                    'Rendimento_%': float(optimizer.mu[t]*100),
                    'Volatilità_%': float(np.sqrt(optimizer.S.loc[t,t])*100),
                    **{f'{n.upper()}_%': optimizer.results[n]['weights'][i]*100 for n in optimizer.results}
                } for i, t in enumerate(optimizer.tickers)]
                
                df_export = pd.DataFrame(rows)
                
                csv = df_export.to_csv(index=False)
                
                st.download_button(
                    label="📥 Scarica CSV",
                    data=csv,
                    file_name=f"portfolio_optimization_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                st.dataframe(df_export, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Errore durante l'ottimizzazione: {str(e)}")
            st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p>📊 Portfolio Optimizer Pro | Powered by PyPortfolioOpt & Streamlit</p>
        <p>⚠️ Disclaimer: Questo tool è solo a scopo educativo. Non costituisce consulenza finanziaria.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

def plot_cumulative
