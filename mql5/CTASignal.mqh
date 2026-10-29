//+------------------------------------------------------------------+
//| CTASignal.mqh                                                     |
//| In-terminal signal generation: load the research price history,  |
//| splice the broker's live daily bars onto it, run CTAEngine, size lots.   |
//| Mirrors live/live_prices.py and live/signal_service.py.          |
//+------------------------------------------------------------------+
#property strict
#include "CTAEngine.mqh"
#include "CTATargets.mqh"

// research column -> broker root, dated?, bucket, PROFIT currency of the broker symbol (for the USD lot value), inverted?
struct CTASpec { string sym; string root; bool dated; int bucket; string ccy; bool inverted; };
int CTA_Specs(CTASpec &s[])
  {
   ArrayResize(s, 14);   // 13 -> 14 on 2026-09-25: +btc +eth, -gilt
   int i = 0;
   s[i].sym="spx";    s[i].root="SPX500";    s[i].dated=true;  s[i].bucket=0; s[i].ccy="USD"; s[i].inverted=false; i++;
   s[i].sym="nasdaq"; s[i].root="NDAQ100";   s[i].dated=true;  s[i].bucket=0; s[i].ccy="USD"; s[i].inverted=false; i++;
   s[i].sym="nikkei"; s[i].root="JPN225";    s[i].dated=false; s[i].bucket=0; s[i].ccy="JPY"; s[i].inverted=false; i++;
   s[i].sym="gold";   s[i].root="GOLD";      s[i].dated=true;  s[i].bucket=1; s[i].ccy="USD"; s[i].inverted=false; i++;
   s[i].sym="silver"; s[i].root="SILVER";    s[i].dated=true;  s[i].bucket=1; s[i].ccy="USD"; s[i].inverted=false; i++;
   s[i].sym="copper"; s[i].root="COPPER";    s[i].dated=true;  s[i].bucket=1; s[i].ccy="USc"; s[i].inverted=false; i++;
   s[i].sym="wti";    s[i].root="USOIL";     s[i].dated=true;  s[i].bucket=1; s[i].ccy="USD"; s[i].inverted=false; i++;
   s[i].sym="brent";  s[i].root="UKOIL";     s[i].dated=true;  s[i].bucket=1; s[i].ccy="USD"; s[i].inverted=false; i++;
   s[i].sym="aud";    s[i].root="AUDUSD.a";  s[i].dated=false; s[i].bucket=2; s[i].ccy="USD"; s[i].inverted=false; i++;
   s[i].sym="gbp";    s[i].root="GBPUSD.a";  s[i].dated=false; s[i].bucket=2; s[i].ccy="USD"; s[i].inverted=false; i++;
   s[i].sym="jpy";    s[i].root="USDJPY.a";  s[i].dated=false; s[i].bucket=2; s[i].ccy="USD"; s[i].inverted=true;  i++;
   s[i].sym="us10y";  s[i].root="TNOTE10YR"; s[i].dated=true;  s[i].bucket=3; s[i].ccy="USD"; s[i].inverted=false; i++;
   // Crypto, added 2026-09-25. Rolling CFDs (dated=false), so no roll logic
   // applies. Bucket 4 is a FIFTH asset class: n_buckets below must be 5.
   // Both carry no history before 2015-01-02 / 2018-02-08; the engine
   // propagates NaN exactly as pandas does, so they hold nothing until then.
   s[i].sym="btc";    s[i].root="BITCOIN";   s[i].dated=false; s[i].bucket=4; s[i].ccy="USD"; s[i].inverted=false; i++;
   s[i].sym="eth";    s[i].root="ETHEREUM";  s[i].dated=false; s[i].bucket=4; s[i].ccy="USD"; s[i].inverted=false; i++;
   // GILT removed 2026-09-25. The broker rejected every order server-side for ten
   // days; the live book now runs the same 14 markets research validated.
   // To restore it: add the row back with bucket=3, add "gilt" to specs.py,
   // and regenerate cta_port_prices.csv (write_ea_history.py) so the header
   // matches, or the check below will stop the EA.
   return i;
  }

// Price-unit currency of a broker root per the specs (copper is quoted in US
// cents, so its lot value is contract x price x 0.01; the terminal's own
// SYMBOL_CURRENCY_PROFIT says USD and would overstate it a hundredfold).
string CTA_SpecCcy(const string root)
  {
   CTASpec sp[]; int k = CTA_Specs(sp);
   for(int i = 0; i < k; i++) if(sp[i].root == root) return sp[i].ccy;
   return "USD";
  }

//+------------------------------------------------------------------+
//| Load the research history (date + one close per column).          |
//+------------------------------------------------------------------+
bool CTA_LoadHistory(const string fname, string &hdr[], datetime &dates[], double &px[], int &n, int &m)
  {
   int h = FileOpen(fname, FILE_READ | FILE_CSV | FILE_ANSI, ',');
   if(h == INVALID_HANDLE) { PrintFormat("CTA: cannot open history %s (%d)", fname, GetLastError()); return false; }
   m = 0; FileReadString(h);
   while(!FileIsLineEnding(h) && !FileIsEnding(h)) { string s = FileReadString(h); ArrayResize(hdr, m + 1); hdr[m++] = s; }
   n = 0;
   while(!FileIsEnding(h))
     {
      string d = FileReadString(h);
      if(d == "") break;
      ArrayResize(dates, n + 1); ArrayResize(px, (n + 1) * m);
      dates[n] = StringToTime(d);
      for(int j = 0; j < m; j++) { string v = FileReadString(h); px[n * m + j] = (v == "") ? CTA_NAN() : StringToDouble(v); }
      n++;
     }
   FileClose(h);
   return n > 0;
  }

//+------------------------------------------------------------------+
//| Write the price matrix in the export's format (date, one close per |
//| column, blank for NaN). Used to append the spliced days to the      |
//| history file so the next run anchors on yesterday: the broker's dated|
//| contracts carry bars only from a few months before they list, so a  |
//| contract born after the export's end would otherwise never overlap  |
//| it and that market would splice to NaN. Keep a copy of the original |
//| export elsewhere; this rewrites the file in place.                  |
//+------------------------------------------------------------------+
bool CTA_WriteHistory(const string fname, const string &hdr[], const datetime &dates[], const double &px[],
                      const int n, const int m)
  {
   int h = FileOpen(fname, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(h == INVALID_HANDLE) { PrintFormat("CTA: cannot write %s (%d)", fname, GetLastError()); return false; }
   string line = "date";
   for(int j = 0; j < m; j++) line += "," + hdr[j];
   FileWrite(h, line);
   for(int t = 0; t < n; t++)
     {
      line = TimeToString(dates[t], TIME_DATE);
      for(int j = 0; j < m; j++)
         line += "," + (CTA_IsNaN(px[t * m + j]) ? "" : DoubleToString(px[t * m + j], 10));
      FileWrite(h, line);
     }
   FileClose(h);
   return true;
  }

datetime CTA_DayOf(const datetime t) { return (datetime)((t / 86400) * 86400); }

//+------------------------------------------------------------------+
//| Extend the history with the broker's completed D1 closes, ratio-spliced  |
//| at the last common date per column. Only dates after the history  |
//| end are added; the union of the symbols' calendars is used.       |
//+------------------------------------------------------------------+
bool CTA_SpliceLiveImpl(const CTASpec &spec[], const int m, datetime &dates[], double &px[], int &n,
                        const int roll_days, const datetime last, const datetime today, string &report)
  {
   // new calendar: union of bar dates in (last, today)
   datetime newd[]; int nn = 0;
   double   closes[]; datetime ctimes[]; int coff[]; int ccount[];   // per column, flat
   ArrayResize(coff, m); ArrayResize(ccount, m);
   int total = 0; int failed = 0;
   report = "";
   for(int j = 0; j < m; j++)
     {
      string sym = spec[j].dated ? CTA_ResolveContract(spec[j].root, roll_days) : spec[j].root;
      coff[j] = total; ccount[j] = 0;
      if(sym == "") { report += spec[j].sym + ": no contract; "; failed++; continue; }
      SymbolSelect(sym, true);
      MqlRates r[];
      int got = CopyRates(sym, PERIOD_D1, last - 20 * 86400, today + 86400, r);
      if(got <= 0) { report += spec[j].sym + ": no bars; "; failed++; continue; }
      for(int i = 0; i < got; i++)
        {
         datetime d = CTA_DayOf(r[i].time);
         if(d >= today) continue;                       // today's bar is still forming
         MqlDateTime dt; TimeToStruct(d, dt);
         if(dt.day_of_week == 0 || dt.day_of_week == 6) continue;
         double c = r[i].close;
         if(spec[j].inverted) c = 1.0 / c;
         ArrayResize(closes, total + 1); ArrayResize(ctimes, total + 1);
         closes[total] = c; ctimes[total] = d; total++; ccount[j]++;
         if(d > last)
           {
            bool seen = false;
            for(int q = 0; q < nn; q++) if(newd[q] == d) { seen = true; break; }
            if(!seen) { ArrayResize(newd, nn + 1); newd[nn++] = d; }
           }
        }
     }
   if(nn == 0) { report += "nothing to add"; return true; }
   ArraySort(newd);
   // 2. grow the matrix
   int n0 = n;
   ArrayResize(dates, n0 + nn); ArrayResize(px, (n0 + nn) * m);
   for(int q = 0; q < nn; q++) { dates[n0 + q] = newd[q]; for(int j = 0; j < m; j++) px[(n0 + q) * m + j] = CTA_NAN(); }
   n = n0 + nn;
   // 3. per column: anchor ratio at the last date both have, then fill the new rows
   for(int j = 0; j < m; j++)
     {
      if(ccount[j] == 0) continue;
      double ratio = CTA_NAN(); datetime anchor = 0;
      for(int t = n0 - 1; t >= 0 && t >= n0 - 40; t--)
        {
         if(CTA_IsNaN(px[t * m + j])) continue;
         datetime d = CTA_DayOf(dates[t]);
         for(int i = 0; i < ccount[j]; i++)
            if(ctimes[coff[j] + i] == d) { ratio = px[t * m + j] / closes[coff[j] + i]; anchor = d; break; }
         if(!CTA_IsNaN(ratio)) break;
        }
      if(CTA_IsNaN(ratio)) { report += spec[j].sym + ": no overlap; "; failed++; continue; }
      int added = 0;
      for(int q = 0; q < nn; q++)
         for(int i = 0; i < ccount[j]; i++)
            if(ctimes[coff[j] + i] == newd[q]) { px[(n0 + q) * m + j] = closes[coff[j] + i] * ratio; added++; break; }
      report += StringFormat("%s +%d (anchor %s ratio %.5f); ", spec[j].sym, added, TimeToString(anchor, TIME_DATE), ratio);
     }
   // A market that could not be extended would carry NaN for the new days,
   // score to nothing, and be FLATTENED at the next rebalance because the
   // feed hiccuped. Refuse the whole run instead; the cached targets stand.
   if(failed > 0) { report = StringFormat("%d market(s) not extended, run refused: ", failed) + report; return false; }
   return true;
  }

//+------------------------------------------------------------------+
//| USD value of one lot of a broker symbol at the current mid.          |
//+------------------------------------------------------------------+
double CTA_UsdPerUnit(const string ccy)
  {
   if(ccy == "USD") return 1.0;
   if(ccy == "USc") return 0.01;
   string pair = ""; bool inv = false;
   if(ccy == "GBP") pair = "GBPUSD.a"; else if(ccy == "AUD") pair = "AUDUSD.a"; else if(ccy == "JPY") { pair = "USDJPY.a"; inv = true; }
   if(pair == "") return 1.0;
   SymbolSelect(pair, true);
   double mid = (SymbolInfoDouble(pair, SYMBOL_BID) + SymbolInfoDouble(pair, SYMBOL_ASK)) / 2.0;
   if(mid <= 0) return CTA_NAN();
   return inv ? 1.0 / mid : mid;
  }

double CTA_LotValueUsd(const string sym, const string ccy)
  {
   SymbolSelect(sym, true);
   double cs = SymbolInfoDouble(sym, SYMBOL_TRADE_CONTRACT_SIZE);
   if(StringSubstr(sym, 0, 6) == "USDJPY") return cs;         // 100k USD per lot
   double mid = (SymbolInfoDouble(sym, SYMBOL_BID) + SymbolInfoDouble(sym, SYMBOL_ASK)) / 2.0;
   if(mid <= 0) return CTA_NAN();
   return cs * mid * CTA_UsdPerUnit(ccy);
  }

//+------------------------------------------------------------------+
//| Run the engine on history + live bars and produce targets.        |
//| equity_leg: S&P notional as a fraction of equity, folded into spx. |
//+------------------------------------------------------------------+
bool CTA_ComputeTargets(const string history_file, const double equity, const double equity_leg,
                        const double stop_atr, const int roll_days, CTAMeta &meta, CTATarget &out[],
                        string &report, const bool persist = true)
  {
   CTASpec spec[]; int ms = CTA_Specs(spec);
   string hdr[]; datetime dates[]; double px[]; int n = 0, m = 0;
   if(!CTA_LoadHistory(history_file, hdr, dates, px, n, m)) return false;
   if(m != ms) { PrintFormat("CTA: history has %d columns, specs %d", m, ms); return false; }
   for(int j = 0; j < m; j++) if(hdr[j] != spec[j].sym) { PrintFormat("CTA: column %d is %s, expected %s", j, hdr[j], spec[j].sym); return false; }
   int n_before = n;
   if(!CTA_SpliceLiveImpl(spec, m, dates, px, n, roll_days, CTA_DayOf(dates[n - 1]), CTA_DayOf(TimeCurrent()), report)) return false;
   // persist: write the spliced rows back into the history file, so tomorrow's
   // splice anchors on today's contract at yesterday's date rather than on the
   // Bloomberg export's last date (which a newly listed contract never overlaps).
   if(persist && n > n_before)
     {
      if(CTA_WriteHistory(history_file, hdr, dates, px, n, m))
         report += StringFormat("appended %d rows to %s (to %s); ", n - n_before, history_file, TimeToString(dates[n - 1], TIME_DATE));
     }

   CTAParams p; CTA_DefaultParams(p);
   int bucket[]; ArrayResize(bucket, m); for(int j = 0; j < m; j++) bucket[j] = spec[j].bucket;
   bool is_rebal[]; CTA_MonthlyRebal(dates, n, is_rebal);
   double w[], lev[], mult[];
   CTA_BuildTargets(px, n, m, dates, bucket, 5, is_rebal, p, w, lev, mult);   // 5 buckets since 2026-09-25 (crypto)

   // rebalance id: month of the last rebalance day at or before the last row
   int tr = n - 1; while(tr > 0 && !is_rebal[tr]) tr--;
   MqlDateTime dt; TimeToStruct(dates[tr], dt);
   meta.rebalance_id = StringFormat("%04d-%02d", dt.year, dt.mon);
   meta.asof = TimeToString(dates[n - 1], TIME_DATE);
   meta.equity = equity;
   meta.engine = "mql5";
   ArrayResize(out, m);
   for(int j = 0; j < m; j++)
     {
      double weight = w[(n - 1) * m + j];
      if(spec[j].sym == "spx") weight += equity_leg;
      string sym = spec[j].dated ? CTA_ResolveContract(spec[j].root, roll_days) : spec[j].root;
      double lv = (sym == "") ? CTA_NAN() : CTA_LotValueUsd(sym, spec[j].ccy);
      double lots = (CTA_IsNaN(lv) || lv <= 0) ? 0.0 : weight * equity / lv;
      if(spec[j].inverted) lots = -lots;
      double v = (sym == "") ? 0.0 : CTA_NormalizeLots(sym, lots);
      out[j].root = spec[j].root; out[j].dated = spec[j].dated;
      out[j].side = (v == 0.0) ? 0 : (lots > 0 ? 1 : -1);
      out[j].lots = v; out[j].weight = weight; out[j].notional = weight * equity; out[j].stop_atr = stop_atr;
     }
   meta.n = m;
   return true;
  }

//+------------------------------------------------------------------+
//| Write the engine's targets in the same format as the Python file. |
//|                                                                    |
//| Field 2 is when this file was WRITTEN, which is what file mode's   |
//| age check wants. Field 7 is the date of the newest PRICE the       |
//| targets were computed from, appended on 2026-10-16. Without it the |
//| archive held no record of the price date at all, so the frozen-    |
//| price failure of September could not be seen after the fact, and   |
//| the manual check the handover prescribed ("look at meta.asof in    |
//| cta_targets_engine.csv") was reading the write time and could      |
//| never have caught it. Appended rather than inserted so the         |
//| positional readers in reconcile_daily.py keep working.             |
//+------------------------------------------------------------------+
void CTA_WriteTargets(const string fname, const CTAMeta &meta, const CTATarget &t[])
  {
   int h = FileOpen(fname, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(h == INVALID_HANDLE) return;
   FileWrite(h, "META", TimeToString(TimeGMT(), TIME_DATE | TIME_SECONDS), meta.rebalance_id,
             DoubleToString(meta.equity, 2), (string)meta.n, meta.engine, meta.asof);
   FileWrite(h, "symbol", "kind", "side", "lots", "weight", "notional_usd", "stop_atr_mult");
   for(int i = 0; i < ArraySize(t); i++)
      FileWrite(h, t[i].root, t[i].dated ? "dated" : "cfd", (string)t[i].side, DoubleToString(t[i].lots, 2),
                DoubleToString(t[i].weight, 5), DoubleToString(t[i].notional, 0), DoubleToString(t[i].stop_atr, 2));
   FileClose(h);
  }
