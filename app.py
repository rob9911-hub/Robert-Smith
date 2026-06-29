from flask import Flask, render_template, jsonify, request
import yfinance as yf
import numpy as np
import math
import os
import requests
from datetime import date

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SEC EDGAR — free 10+ year financial history for US filers
# ─────────────────────────────────────────────────────────────────────────────
# SEC asks for a contact email in the User-Agent. Set SEC_CONTACT_EMAIL env var
# (e.g. in Render) so a personal address isn't committed to a public repo.
SEC_CONTACT = os.environ.get('SEC_CONTACT_EMAIL', 'contact@example.com')
SEC_HEADERS = {'User-Agent': f'StockAnalyzer/1.0 {SEC_CONTACT}'}
_TICKER_CIK = {}   # cache: ticker -> 10-digit CIK string

def load_ticker_cik():
    global _TICKER_CIK
    if _TICKER_CIK:
        return _TICKER_CIK
    try:
        r = requests.get('https://www.sec.gov/files/company_tickers.json',
                         headers=SEC_HEADERS, timeout=15)
        r.raise_for_status()
        for row in r.json().values():
            _TICKER_CIK[row['ticker'].upper()] = str(row['cik_str']).zfill(10)
    except Exception:
        pass
    return _TICKER_CIK

def get_edgar_facts(ticker):
    cik = load_ticker_cik().get(ticker.upper())
    if not cik:
        return None
    try:
        r = requests.get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json',
                         headers=SEC_HEADERS, timeout=20)
        r.raise_for_status()
        return r.json().get('facts', {})
    except Exception:
        return None

def edgar_series(facts, concepts, flow=True):
    """Annual (year, value) pairs oldest→newest from EDGAR XBRL facts.
    flow=True  → income/cash-flow items (need ~365-day duration).
    flow=False → balance-sheet point-in-time items (from 10-K filings)."""
    combined = {}
    for concept in concepts:                      # priority order: first wins per year
        node = facts.get('us-gaap', {}).get(concept) or facts.get('dei', {}).get(concept)
        if not node:
            continue
        units = node.get('units', {})
        # Only trust USD (monetary) or share-count data; skip foreign-currency filers
        arr = units.get('USD') or units.get('shares') or []
        for item in arr:
            val, end = item.get('val'), item.get('end')
            if val is None or not end:
                continue
            if '10-K' not in item.get('form', '') and item.get('fp') != 'FY':
                continue
            if flow:
                start = item.get('start')
                if not start:
                    continue
                try:
                    days = (date.fromisoformat(end) - date.fromisoformat(start)).days
                except Exception:
                    continue
                if not (350 <= days <= 380):       # keep full-year periods only
                    continue
            yr = int(end[:4])
            if yr not in combined:
                combined[yr] = float(val)
    return sorted(combined.items())

def get_edgar_data(ticker):
    """Return a dict of 10+ year series from EDGAR, or None if unavailable."""
    facts = get_edgar_facts(ticker)
    if not facts:
        return None

    revenues = edgar_series(facts, [
        'RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues',
        'RevenueFromContractWithCustomerIncludingAssessedTax', 'SalesRevenueNet'])
    if len(revenues) < 5:        # not enough history → let yfinance handle it
        return None

    net_incomes   = edgar_series(facts, ['NetIncomeLoss'])
    gross_profits = edgar_series(facts, ['GrossProfit'])
    op_cfs = edgar_series(facts, [
        'NetCashProvidedByUsedInOperatingActivities',
        'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations'])
    capexs = edgar_series(facts, [
        'PaymentsToAcquirePropertyPlantAndEquipment', 'PaymentsToAcquireProductiveAssets'])
    equity = edgar_series(facts, [
        'StockholdersEquity',
        'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'], flow=False)
    dividends = edgar_series(facts, ['PaymentsOfDividendsCommonStock', 'PaymentsOfDividends'])
    shares = edgar_series(facts, [
        'WeightedAverageNumberOfDilutedSharesOutstanding',
        'WeightedAverageNumberOfSharesOutstandingBasic'])
    acquisitions = edgar_series(facts, [
        'PaymentsToAcquireBusinessesNetOfCashAcquired',
        'PaymentsToAcquireBusinessesAndInterestInAffiliates',
        'PaymentsToAcquireBusinessesGross'])

    # FCF = operating CF - capex (matched by year)
    capex_map = dict(capexs)
    fcfs = [(yr, ocf - abs(capex_map.get(yr, 0))) for yr, ocf in op_cfs]
    shares_map = dict(shares)

    return {
        'revenues': revenues, 'net_incomes': net_incomes, 'gross_profits': gross_profits,
        'fcfs': fcfs, 'equity': equity, 'dividends': dividends,
        'shares_series': shares, 'shares_map': shares_map, 'acquisitions': acquisitions,
    }

def detect_restructuring(yf_series, edgar_series_):
    """True if EDGAR and yfinance revenue disagree in overlapping years — a sign
    the company spun off/restated (e.g. GE), making EDGAR's long as-filed history
    not comparable to today's entity. yfinance carries the restated figures."""
    yf_map = {y: v for y, v in yf_series}
    ed_map = {y: v for y, v in edgar_series_}
    common = sorted(set(yf_map) & set(ed_map))
    if len(common) < 2:
        return False
    for y in common:
        a, b = ed_map[y], yf_map[y]
        if a and b and abs(b) > 1:
            ratio = a / b
            if ratio < 0.7 or ratio > 1.43:   # >~43% mismatch on the same year
                return True
    return False

@app.after_request
def add_cors(resp):
    # Allow the page to call the API even when opened from file:// or another origin
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return resp

def safe(info, key, default=None):
    v = info.get(key)
    return v if v is not None else default

def clean(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj

def cagr(start, end, years):
    if start and end and start > 0 and years > 0:
        return ((end / start) ** (1 / years) - 1) * 100
    return None

def fmt_large(n):
    if n is None or (isinstance(n, float) and (math.isnan(n) or math.isinf(n))):
        return 'N/A'
    sign = '-' if n < 0 else ''
    a = abs(n)
    if a >= 1e12: return f'{sign}${a/1e12:.2f}T'
    if a >= 1e9:  return f'{sign}${a/1e9:.2f}B'
    if a >= 1e6:  return f'{sign}${a/1e6:.2f}M'
    return f'{sign}${a:.0f}'

def get_usd_rate(currency):
    if not currency or currency == 'USD': return 1.0
    try:
        fx = yf.Ticker(f'{currency}USD=X')
        rate = fx.info.get('regularMarketPrice') or fx.info.get('previousClose')
        if rate: return float(rate)
    except: pass
    fallbacks = {'CNY': 0.138, 'HKD': 0.128, 'EUR': 1.08, 'GBP': 1.27,
                 'JPY': 0.0067, 'KRW': 0.00073, 'INR': 0.012, 'CAD': 0.74,
                 'AUD': 0.65, 'CHF': 1.12, 'TWD': 0.031, 'BRL': 0.18}
    return fallbacks.get(currency, 1.0)

def extract_row(df, names):
    for name in names:
        if df is not None and not df.empty and name in df.index:
            row = df.loc[name].dropna()
            pairs = []
            for d, v in zip(row.index, row.values):
                try:
                    fv = float(v)
                    if not math.isnan(fv):
                        pairs.append((str(d.year), fv))
                except: pass
            return sorted(pairs, key=lambda x: x[0])
    return []

def get_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    info  = stock.info

    if not info.get('regularMarketPrice') and not info.get('currentPrice') and not info.get('marketCap'):
        raise ValueError(f"No data found for ticker: {ticker_symbol}")

    income   = stock.income_stmt
    cashflow = stock.cashflow
    balance  = stock.balance_sheet
    q_income   = stock.quarterly_income_stmt
    q_cashflow = stock.quarterly_cashflow

    current_price      = safe(info, 'currentPrice') or safe(info, 'regularMarketPrice', 0)
    market_cap         = safe(info, 'marketCap', 0) or 0
    enterprise_value   = safe(info, 'enterpriseValue', 0) or 0
    shares_outstanding = safe(info, 'sharesOutstanding', 1) or 1
    total_debt         = safe(info, 'totalDebt', 0) or 0
    total_cash         = safe(info, 'totalCash', 0) or 0

    # Balance-sheet items for EM-style Enterprise Value (uses TOTAL LIABILITIES)
    def bs_val(names):
        if balance is not None and not balance.empty:
            for nm in names:
                if nm in balance.index:
                    row = balance.loc[nm].dropna()
                    if not row.empty:
                        return float(row.iloc[0])
        return None
    total_liabilities = bs_val(['Total Liabilities Net Minority Interest', 'Total Liabilities']) or total_debt
    cash_equiv        = bs_val(['Cash And Cash Equivalents']) or 0
    cash_st_inv       = bs_val(['Cash Cash Equivalents And Short Term Investments']) or total_cash

    fin_currency = safe(info, 'financialCurrency', 'USD')
    fx_rate      = get_usd_rate(fin_currency)

    # ── True trailing-twelve-months (sum of last 4 quarters) — matches EM ──
    def sum_last4(df, names):
        for nm in names:
            if df is not None and not df.empty and nm in df.index:
                vals = df.loc[nm].dropna()
                if len(vals) >= 4:
                    return float(vals.iloc[:4].sum()) * fx_rate
        return None
    ttm_revenue = sum_last4(q_income, ['Total Revenue', 'Revenue'])
    ttm_ni      = sum_last4(q_income, ['Net Income', 'Net Income Common Stockholders'])
    ttm_gross   = sum_last4(q_income, ['Gross Profit'])
    ttm_opcf    = sum_last4(q_cashflow, ['Operating Cash Flow', 'Total Cash From Operating Activities'])
    ttm_capex   = sum_last4(q_cashflow, ['Capital Expenditure', 'Capital Expenditures'])
    ttm_fcf     = (ttm_opcf - abs(ttm_capex)) if (ttm_opcf is not None and ttm_capex is not None) else None
    ttm_dividends = sum_last4(q_cashflow, ['Cash Dividends Paid', 'Common Stock Dividend Paid', 'Payment Of Dividends'])

    def extract_usd(df, names):
        return [(yr, v * fx_rate) for yr, v in extract_row(df, names)]

    revenues      = extract_usd(income,  ['Total Revenue', 'Revenue'])
    net_incomes   = extract_usd(income,  ['Net Income', 'Net Income Common Stockholders'])
    gross_profits = extract_usd(income,  ['Gross Profit'])
    shares_history = extract_row(balance, ['Ordinary Shares Number', 'Share Issued'])
    book_values   = extract_usd(balance, ['Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity'])

    # FCF = operating CF - capex
    fcfs = []
    if cashflow is not None and not cashflow.empty:
        opcf_row = capex_row = None
        for n in ['Operating Cash Flow', 'Total Cash From Operating Activities', 'Cash Flow From Continuing Operating Activities']:
            if n in cashflow.index: opcf_row = cashflow.loc[n]; break
        for n in ['Capital Expenditure', 'Capital Expenditures', 'Purchase Of PPE']:
            if n in cashflow.index: capex_row = cashflow.loc[n]; break
        if opcf_row is not None:
            for d in opcf_row.index:
                try:
                    opcf = float(opcf_row[d])
                    if math.isnan(opcf): continue
                    capex = 0
                    if capex_row is not None:
                        cv = float(capex_row[d]) if d in capex_row.index else 0
                        capex = 0 if math.isnan(cv) else cv
                    fcfs.append((str(d.year), (opcf - abs(capex)) * fx_rate))
                except: pass
            fcfs = sorted(fcfs, key=lambda x: x[0])

    # Dividends paid — TTM (last 4 quarters) to match EM; fall back to latest annual
    dividends_paid = abs(ttm_dividends) if ttm_dividends is not None else 0
    if not dividends_paid and cashflow is not None and not cashflow.empty:
        for name in ['Payment Of Dividends', 'Common Stock Dividend Paid', 'Cash Dividends Paid']:
            if name in cashflow.index:
                v = cashflow.loc[name].dropna()
                if not v.empty: dividends_paid = abs(float(v.iloc[0])) * fx_rate
                break

    # Acquisitions over last 5yr — yfinance tags large M&A cleanly here
    # (e.g. MSFT 'Purchase Of Business' FY2024 = -$69B Activision); EDGAR's
    # XBRL tagging for big deals is inconsistent, so we use yfinance.
    acquisitions_5yr = 0
    if cashflow is not None and not cashflow.empty:
        for name in ['Purchase Of Business', 'Net Business Purchase And Sale',
                     'Acquisitions Net', 'Acquisition Of Business']:
            if name in cashflow.index:
                vals = cashflow.loc[name].dropna()
                acquisitions_5yr = -abs(float(vals.iloc[:5].sum())) * fx_rate
                break

    # ── SEC EDGAR override: replace short yfinance series with 10+ yr history ──
    # But first reconcile against yfinance: if the two disagree in overlapping
    # years, the company restructured (spinoff/restatement, e.g. GE Aerospace) and
    # EDGAR's as-filed history is NOT comparable to today's entity — so we keep
    # yfinance's restated (continuing-operations) series instead.
    data_source = 'Yahoo Finance'
    bv_is_per_share = False     # yfinance stores equity ($); EDGAR stores BVPS
    edgar = None
    try:
        edgar = get_edgar_data(ticker_symbol)
    except Exception:
        edgar = None
    if edgar:
        ed_rev = [(str(y), v) for y, v in edgar['revenues']]
        restructured = detect_restructuring(revenues, ed_rev)
        if restructured:
            data_source = 'Yahoo Finance (restated)'
        else:
            data_source = 'SEC EDGAR'
            revenues      = ed_rev
            net_incomes   = [(str(y), v) for y, v in edgar['net_incomes']]   or net_incomes
            gross_profits = [(str(y), v) for y, v in edgar['gross_profits']] or gross_profits
            if edgar['fcfs']:
                fcfs = [(str(y), v) for y, v in edgar['fcfs']]
            # Book value per share = equity / that year's diluted share count
            bvps = []
            for y, eq in edgar['equity']:
                sh = edgar['shares_map'].get(y)
                if sh and sh > 0:
                    bvps.append((str(y), eq / sh))
            if bvps:
                book_values = bvps
                bv_is_per_share = True
            if edgar['shares_series']:
                shares_history = [(str(y), v) for y, v in edgar['shares_series']]
            # dividends_paid already set from TTM (last 4 quarters) above

    # Moving averages + 52wk high/low + ATH (with dates) from price history
    week52_high = week52_low = ath = None
    week52_high_date = week52_low_date = ath_date = None
    try:
        hist = stock.history(period='2y')
        cl   = hist['Close']
        ma_25  = float(cl.tail(25).mean())  if len(cl) >= 25  else None
        ma_50  = float(cl.tail(50).mean())  if len(cl) >= 50  else safe(info, 'fiftyDayAverage')
        ma_100 = float(cl.tail(100).mean()) if len(cl) >= 100 else None
        ma_200 = float(cl.tail(200).mean()) if len(cl) >= 200 else safe(info, 'twoHundredDayAverage')

        # 52-week high/low from last 252 trading days
        last_year = hist.tail(252)
        if not last_year.empty:
            hi_idx = last_year['High'].idxmax()
            lo_idx = last_year['Low'].idxmin()
            week52_high = float(last_year['High'].max())
            week52_low  = float(last_year['Low'].min())
            week52_high_date = hi_idx.strftime('%m/%d/%y')
            week52_low_date  = lo_idx.strftime('%m/%d/%y')

        # All-time high from full history
        full = stock.history(period='max')
        if not full.empty:
            ath_idx  = full['High'].idxmax()
            ath      = float(full['High'].max())
            ath_date = ath_idx.strftime('%m/%d/%y')
    except:
        ma_25 = ma_100 = None
        ma_50  = safe(info, 'fiftyDayAverage')
        ma_200 = safe(info, 'twoHundredDayAverage')
        week52_high = safe(info, 'fiftyTwoWeekHigh')
        week52_low  = safe(info, 'fiftyTwoWeekLow')

    # ── Core TTM values (true trailing-twelve-months — what EM displays) ──
    last_revenue    = ttm_revenue if ttm_revenue is not None else \
                      ((safe(info, 'totalRevenue', 0) or 0) * fx_rate) or (revenues[-1][1] if revenues else 0)
    last_net_income = ttm_ni if ttm_ni is not None else \
                      ((safe(info, 'netIncomeToCommon', 0) or 0) * fx_rate) or (net_incomes[-1][1] if net_incomes else 0)
    # FCF TTM: computed OpCF-CapEx (yfinance's freeCashflow field is unreliable)
    last_fcf = ttm_fcf if ttm_fcf is not None else (fcfs[-1][1] if fcfs else 0)

    # Windowed averages over the n most-recent periods = [TTM] + prior fiscal years.
    # EM includes the current TTM as the latest point, so we do too.
    def window_avg(series, ttm_val, n):
        annual_newest_first = [v for _, v in series][::-1]
        pts = ([ttm_val] if ttm_val is not None else []) + annual_newest_first
        pts = [p for p in pts if p is not None][:n]
        return float(np.mean(pts)) if pts else None

    avg_ni_5yr   = window_avg(net_incomes, last_net_income, 5) or 0
    avg_fcf_5yr  = window_avg(fcfs,        last_fcf,        5) or (last_fcf or 0)
    avg_rev_5yr  = window_avg(revenues,    last_revenue,    5)
    avg_ni_10yr  = window_avg(net_incomes, last_net_income, 10)
    avg_rev_10yr = window_avg(revenues,    last_revenue,    10)
    avg_fcf_10yr = window_avg(fcfs,        last_fcf,        10)

    # FCF margins (1yr/5yr/10yr) for the Stock Analyzer historical columns
    fcf_margin_ttm  = (last_fcf / last_revenue * 100) if last_revenue else None
    fcf_margin_5yr  = (avg_fcf_5yr / avg_rev_5yr * 100) if (avg_fcf_5yr and avg_rev_5yr) else None
    fcf_margin_10yr = (avg_fcf_10yr / avg_rev_10yr * 100) if (avg_fcf_10yr and avg_rev_10yr) else None

    # Margins
    gross_margin = (ttm_gross / ttm_revenue * 100) if (ttm_gross and ttm_revenue) else None
    if gross_margin is None:
        gm = safe(info, 'grossMargins')
        gross_margin = gm * 100 if gm else (
            gross_profits[-1][1] / last_revenue * 100 if gross_profits and last_revenue else None)

    profit_margin_ttm = (last_net_income / last_revenue * 100) if last_revenue else None

    # Period profit margins = avg NI / avg revenue over the window (TTM-inclusive).
    # Require enough annual history so the window is genuinely n periods deep.
    profit_margin_5yr = profit_margin_10yr = None
    if avg_ni_5yr and avg_rev_5yr and len(revenues) >= 4:
        profit_margin_5yr = avg_ni_5yr / avg_rev_5yr * 100
    if avg_ni_10yr and avg_rev_10yr and len(revenues) >= 9:
        profit_margin_10yr = avg_ni_10yr / avg_rev_10yr * 100

    # Equity series (newest-first): prefer EDGAR's 10+ yr history, else yfinance
    if edgar and edgar.get('equity'):
        equities = [v for _, v in edgar['equity']][::-1]   # EDGAR oldest→newest, reverse
    else:
        equities = []
        if balance is not None and not balance.empty:
            for n in ['Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity']:
                if n in balance.index:
                    equities = [float(v) * fx_rate for v in balance.loc[n].dropna().values]
                    break
    latest_equity = equities[0] if equities else None

    # ROE = TTM net income / latest equity (matches EM exactly)
    roe = (last_net_income / latest_equity * 100) if (latest_equity and latest_equity > 0) else \
          ((safe(info, 'returnOnEquity') or 0) * 100 or None)

    # ROIC TTM + 5yr + 10yr (net income / invested capital = equity + debt).
    # NOTE: EM's published ROIC uses a proprietary invested-capital definition we
    # can't fully reproduce from filings; this is the standard textbook approximation.
    roic_ttm = roic_5yr = roic_10yr = None
    if latest_equity and last_net_income:
        ic = latest_equity + total_debt
        if ic > 0: roic_ttm = last_net_income / ic * 100
    if equities and avg_ni_5yr:
        avg_eq5 = float(np.mean(equities[:5]))
        if avg_eq5 + total_debt > 0: roic_5yr = avg_ni_5yr / (avg_eq5 + total_debt) * 100
    if len(equities) >= 9 and avg_ni_10yr:
        avg_eq10 = float(np.mean(equities[:10]))
        if avg_eq10 + total_debt > 0: roic_10yr = avg_ni_10yr / (avg_eq10 + total_debt) * 100

    # Ratios
    pe_ttm     = safe(info, 'trailingPE')
    forward_pe = safe(info, 'forwardPE')
    peg_trailing = safe(info, 'trailingPegRatio')
    peg_forward  = safe(info, 'pegRatio')
    ps_ratio = market_cap / last_revenue if market_cap and last_revenue else None
    p_fcf_ttm = market_cap / last_fcf    if last_fcf    and last_fcf    > 0 else None
    p_fcf_5yr = market_cap / avg_fcf_5yr if avg_fcf_5yr and avg_fcf_5yr > 0 else None

    # 5yr P/E
    pe_5yr = None
    if avg_ni_5yr and shares_outstanding and current_price:
        eps_5yr = avg_ni_5yr / shares_outstanding
        if eps_5yr > 0: pe_5yr = current_price / eps_5yr

    # PEG (Past 5yr): P/E TTM / 5yr earnings growth rate. N/A if earnings shrank.
    peg_past5 = None
    if len(net_incomes) >= 2 and pe_ttm:
        ni_start = net_incomes[max(0, len(net_incomes)-6)][1]
        ni_end   = net_incomes[-1][1]
        if ni_start and ni_start > 0 and ni_end > 0:
            yrs = min(5, len(net_incomes)-1)
            eps_growth = cagr(ni_start, ni_end, yrs)
            if eps_growth and eps_growth > 0:
                peg_past5 = pe_ttm / eps_growth

    # Dividends
    div_rate      = safe(info, 'dividendRate')
    div_yield_raw = safe(info, 'dividendYield')
    if div_rate and current_price:
        div_yield = div_rate / current_price * 100
        fwd_div_yield = div_yield
    elif div_yield_raw is not None:
        div_yield = div_yield_raw * 100 if div_yield_raw < 0.5 else div_yield_raw
        fwd_div_yield = div_yield
    else:
        div_yield = fwd_div_yield = None

    # EM-style Enterprise Value uses TOTAL LIABILITIES (not just debt):
    #   Traditional = mktcap + total liabilities - cash & equivalents
    #   Paul's      = mktcap + total liabilities - cash & short-term investments
    enterprise_value = market_cap + total_liabilities - cash_equiv
    pauls_ev         = market_cap + total_liabilities - cash_st_inv

    # EV ratios use Paul's EV (matches EM's displayed EV/Earnings, EV/FCF, etc.)
    ev_fcf          = pauls_ev / last_fcf        if last_fcf        and last_fcf        > 0 else None
    ev_5yr_fcf      = pauls_ev / avg_fcf_5yr     if avg_fcf_5yr     and avg_fcf_5yr     > 0 else None
    ev_earnings     = pauls_ev / last_net_income if last_net_income and last_net_income > 0 else None
    ev_5yr_earnings = pauls_ev / avg_ni_5yr      if avg_ni_5yr      and avg_ni_5yr      > 0 else None

    roa = (safe(info, 'returnOnAssets') or 0) * 100 or None
    # roe already computed above (TTM NI / latest equity)

    # Book value/share CAGR — require the full window, else N/A.
    # EDGAR series is already per-share; yfinance series is total equity.
    bv_cagr_5yr = bv_cagr_10yr = None
    if book_values and (bv_is_per_share or shares_outstanding):
        if bv_is_per_share:
            bv_ps = [(yr, v) for yr, v in book_values]
        else:
            bv_ps = [(yr, v / shares_outstanding) for yr, v in book_values]
        recent = bv_ps[-1][1]
        if len(bv_ps) >= 6:
            bv_cagr_5yr = cagr(bv_ps[-6][1], recent, 5)
        if len(bv_ps) >= 11:
            bv_cagr_10yr = cagr(bv_ps[-11][1], recent, 10)

    # Revenue CAGR by period. EM uses ROLLING TTM: current TTM vs the TTM from N
    # years ago. We approximate the "N years ago" point by interpolating the annual
    # series at the fractional position that matches how far TTM extends past the
    # latest fiscal year (ttm_frac), which reproduces EM's numbers closely.
    ttm_frac = 0.0
    try:
        if q_income is not None and not q_income.empty and income is not None and not income.empty:
            ttm_frac = max(0.0, min(1.0, (q_income.columns[0] - income.columns[0]).days / 365.0))
    except Exception:
        ttm_frac = 0.0

    def interp_revenue(pos):
        if pos < 0 or not revenues:
            return None
        lo = int(math.floor(pos)); hi = lo + 1; frac = pos - lo
        if lo >= len(revenues):
            return None
        if hi >= len(revenues):
            return revenues[lo][1]
        return revenues[lo][1] + frac * (revenues[hi][1] - revenues[lo][1])

    def rev_cagr_n(n):
        if not revenues or not last_revenue:
            return None
        start_val = interp_revenue((len(revenues) - 1) + ttm_frac - n)
        if start_val and start_val > 0:
            return cagr(start_val, last_revenue, n)
        return None

    rev_cagr_1  = rev_cagr_n(1)
    rev_cagr_3  = rev_cagr_n(3)
    rev_cagr_5  = rev_cagr_n(5)
    rev_cagr_10 = rev_cagr_n(10)
    years_of_data = len(revenues)

    # 5yr absolute growth = TTM value minus the value 5 years ago (annual)
    def growth_abs(series, ttm_val):
        if len(series) >= 5 and ttm_val is not None:
            return ttm_val - series[-5][1]
        if len(series) >= 2:
            return series[-1][1] - series[max(0, len(series)-6)][1]
        return None

    fcf_growth_5yr_abs = growth_abs(fcfs,        last_fcf)
    ni_growth_5yr_abs  = growth_abs(net_incomes, last_net_income)
    rev_growth_5yr_abs = growth_abs(revenues,    last_revenue)

    shares_pct = None
    if len(shares_history) >= 2:
        shares_pct = (shares_history[-1][1] - shares_history[0][1]) / shares_history[0][1] * 100

    ltl_5yr_fcf = total_debt / avg_fcf_5yr if total_debt and avg_fcf_5yr and avg_fcf_5yr > 0 else None

    pillars = build_pillars(
        market_cap=market_cap, last_revenue=last_revenue, pe_5yr=pe_5yr,
        roic_5yr=roic_5yr if roic_5yr is not None else roic_ttm, shares_pct=shares_pct,
        fcf_growth_5yr_abs=fcf_growth_5yr_abs, ni_growth_5yr_abs=ni_growth_5yr_abs,
        rev_growth_5yr_abs=rev_growth_5yr_abs, ltl_5yr_fcf=ltl_5yr_fcf, p_fcf_5yr=p_fcf_5yr,
    )

    return {
        'ticker': ticker_symbol.upper(),
        'name': safe(info, 'longName', ticker_symbol.upper()),
        'sector': safe(info, 'sector', 'N/A'),
        'industry': safe(info, 'industry', 'N/A'),
        'data_source': data_source,
        'fin_currency': fin_currency, 'fx_rate': fx_rate,
        'years_of_data': years_of_data,
        'current_price': current_price, 'market_cap': market_cap,
        'enterprise_value': enterprise_value, 'pauls_ev': pauls_ev,
        'shares_outstanding': shares_outstanding,
        'last_revenue': last_revenue, 'last_net_income': last_net_income,
        'avg_ni_5yr': avg_ni_5yr, 'last_fcf': last_fcf, 'avg_fcf_5yr': avg_fcf_5yr,
        'pe_ttm': pe_ttm, 'pe_5yr': pe_5yr, 'forward_pe': forward_pe,
        'ps_ratio': ps_ratio, 'peg_past5': peg_past5, 'peg_forward': peg_forward,
        'p_fcf_ttm': p_fcf_ttm, 'p_fcf_5yr': p_fcf_5yr,
        'gross_margin': gross_margin, 'profit_margin_ttm': profit_margin_ttm,
        'profit_margin_5yr': profit_margin_5yr, 'profit_margin_10yr': profit_margin_10yr,
        'roa': roa, 'roe': roe, 'roic_ttm': roic_ttm, 'roic_5yr': roic_5yr, 'roic_10yr': roic_10yr,
        'div_yield': div_yield, 'fwd_div_yield': fwd_div_yield, 'dividends_paid': dividends_paid,
        'total_debt': total_debt, 'total_cash': total_cash, 'ltl_5yr_fcf': ltl_5yr_fcf,
        'acquisitions_5yr': acquisitions_5yr,
        'ev_fcf': ev_fcf, 'ev_5yr_fcf': ev_5yr_fcf,
        'ev_earnings': ev_earnings, 'ev_5yr_earnings': ev_5yr_earnings,
        'ma_25': ma_25, 'ma_50': ma_50, 'ma_100': ma_100, 'ma_200': ma_200,
        'bv_cagr_5yr': bv_cagr_5yr, 'bv_cagr_10yr': bv_cagr_10yr,
        'rev_cagr_1': rev_cagr_1, 'rev_cagr_3': rev_cagr_3, 'rev_cagr_5': rev_cagr_5, 'rev_cagr_10': rev_cagr_10,
        'fcf_margin_ttm': fcf_margin_ttm, 'fcf_margin_5yr': fcf_margin_5yr, 'fcf_margin_10yr': fcf_margin_10yr,
        'rev_growth_5yr_abs': rev_growth_5yr_abs,
        'ni_growth_5yr_abs': ni_growth_5yr_abs,
        'fcf_growth_5yr_abs': fcf_growth_5yr_abs,
        'shares_pct': shares_pct,
        'week52_high': week52_high, 'week52_low': week52_low, 'ath': ath,
        'week52_high_date': week52_high_date, 'week52_low_date': week52_low_date, 'ath_date': ath_date,
        'revenues': revenues, 'net_incomes': net_incomes,
        'fcfs': fcfs, 'shares_history': shares_history,
        'pillars': pillars,
    }

def build_pillars(market_cap, last_revenue, pe_5yr, roic_5yr, shares_pct,
                  fcf_growth_5yr_abs, ni_growth_5yr_abs, rev_growth_5yr_abs,
                  ltl_5yr_fcf, p_fcf_5yr):
    def p(name, threshold, value_str, passes, group):
        return {'name': name, 'threshold': threshold, 'value': value_str, 'pass': passes, 'group': group}
    return [
        # My Pillars (customizable size/quality gates)
        p('Market Capitalization', '< 1.00T', fmt_large(market_cap),   market_cap < 1e12   if market_cap   else None, 'my'),
        p('Revenue',               '> 21.00B', fmt_large(last_revenue), last_revenue > 21e9 if last_revenue else None, 'my'),
        # EM Pillars (the 8 core quality/value checks)
        p('5YR P/E Ratio',         '< 22.5',  f'{pe_5yr:.2f}'                if pe_5yr else 'N/A',       pe_5yr < 22.5           if pe_5yr is not None else None, 'em'),
        p('5YR ROIC',              '> 9%',    f'{roic_5yr:.2f}%'             if roic_5yr else 'N/A',     roic_5yr > 9            if roic_5yr is not None else None, 'em'),
        p('Shares Outstanding',    '↓',       f'{shares_pct:+.2f}%'          if shares_pct is not None else 'N/A', shares_pct < 0 if shares_pct is not None else None, 'em'),
        p('Cash Flow Growth 5 Yr', '> $0',    fmt_large(fcf_growth_5yr_abs), fcf_growth_5yr_abs > 0     if fcf_growth_5yr_abs is not None else None, 'em'),
        p('Net Income Growth 5 Yr','> $0',    fmt_large(ni_growth_5yr_abs),  ni_growth_5yr_abs > 0      if ni_growth_5yr_abs  is not None else None, 'em'),
        p('Revenue Growth 5 Yr',   '> $0',    fmt_large(rev_growth_5yr_abs), rev_growth_5yr_abs > 0     if rev_growth_5yr_abs is not None else None, 'em'),
        p('LTL / 5 Yr FCF',        '< 5',     f'{ltl_5yr_fcf:.2f}'           if ltl_5yr_fcf is not None else 'N/A', ltl_5yr_fcf < 5 if ltl_5yr_fcf is not None else None, 'em'),
        p('5 YR Price to FCF',     '< 22.5',  f'{p_fcf_5yr:.2f}'             if p_fcf_5yr is not None else 'N/A',   p_fcf_5yr < 22.5 if p_fcf_5yr is not None else None, 'em'),
    ]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze/<ticker>')
def analyze(ticker):
    try:
        data = clean(get_data(ticker.strip().upper()))
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

def value_per_share(rev0, shares, rg, margin, multiple, discount, years):
    """EM's valuation: sum of discounted yearly flows (revenue×margin) for each
    year, PLUS the discounted terminal value (final-year flow × exit multiple)."""
    if not shares:
        return 0.0
    interim = sum(rev0 * ((1 + rg) ** t) * margin / shares / ((1 + discount) ** t)
                  for t in range(1, years + 1))
    terminal = rev0 * ((1 + rg) ** years) * margin * multiple / shares / ((1 + discount) ** years)
    return interim + terminal

def price_irr(rev0, shares, rg, margin, multiple, cp, years):
    """Current Price Return = the annual return (IRR) such that the model value
    equals today's price. Solve value_per_share(rate)=cp by bisection."""
    if not shares or not cp or cp <= 0:
        return None
    lo, hi = -0.5, 2.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if value_per_share(rev0, shares, rg, margin, multiple, mid, years) > cp:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

@app.route('/api/dcf', methods=['POST'])
def dcf():
    """EM-style Stock Analyzer. For each scenario, value the business two ways and
    discount back at the desired return:
      • Multiple of Earnings Value — uses profit margin × P/E
      • Discounted Cash Flow Value — uses FCF margin × P/FCF
    Each = sum of discounted yearly flows + discounted terminal value.
    Current Price Return = the IRR earned buying at today's price."""
    try:
        p = request.json
        cp     = float(p['current_price'])
        shares = float(p['shares_outstanding'])
        rev0   = float(p['last_revenue'])
        years  = int(p.get('years', 10))
        out = {}
        for s in ['low', 'mid', 'high']:
            rg = float(p[f'rev_growth_{s}'])    / 100
            pm = float(p[f'profit_margin_{s}']) / 100
            fm = float(p[f'fcf_margin_{s}'])    / 100
            pe = float(p[f'pe_{s}'])
            pf = float(p[f'pfcf_{s}'])
            dr = float(p[f'desired_return_{s}'])/ 100

            earn_value = value_per_share(rev0, shares, rg, pm, pe, dr, years)
            dcf_value  = value_per_share(rev0, shares, rg, fm, pf, dr, years)

            # Current Price Return: IRR of the FCF-based model at today's price
            # (EM shows one return row; FCF basis is the cash-flow view).
            irr = price_irr(rev0, shares, rg, fm, pf, cp, years)

            out[s] = {
                'earnings_value': round(earn_value, 2),
                'dcf_value': round(dcf_value, 2),
                'price_return': round(irr * 100, 2) if irr is not None else None,
                'earnings_upside': round((earn_value / cp - 1) * 100, 1) if cp else None,
                'dcf_upside': round((dcf_value / cp - 1) * 100, 1) if cp else None,
            }
        return jsonify({'success': True, 'result': clean(out)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    # Local dev server. In production, gunicorn imports `app:app` instead (see Procfile).
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, port=port)
