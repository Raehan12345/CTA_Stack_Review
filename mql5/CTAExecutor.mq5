//+------------------------------------------------------------------+
//| CTAExecutor.mq5                                                   |
//| Executor for the CTA trend sleeve. In engine mode it runs the      |
//| strategy itself (CTAEngine.mqh on the research history plus the broker's  |
//| live bars); in file mode it reads cta_targets.csv from the Python  |
//| service. Either way it reconciles the account's positions once per |
//| rebalance, rolls dated contracts, trails a wide ATR stop rail and  |
//| logs every fill.                                                   |
//+------------------------------------------------------------------+
#property copyright ""
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include "CTASignal.mqh"

enum CTA_MODE { CTA_MODE_ENGINE = 0, CTA_MODE_FILE = 1 };
input CTA_MODE InpMode         = CTA_MODE_ENGINE;     // engine: compute targets in-terminal; file: read Python's
input string  InpHistoryFile   = "cta_port_prices.csv"; // research price history (engine mode)
input bool    InpPersistHist   = true;                // append the broker's spliced days to the history file daily
input double  InpCapital       = 0.0;                 // capital the strategy sizes on, account currency; 0 = the account's equity
input double  InpEquityLeg     = 1.0;                 // S&P notional as a fraction of that capital (engine mode)
input double  InpStopAtr       = 10.0;                // stop rail width in D1 ATRs, 0 = off (engine mode)
input int     InpEngineHourUTC = 22;                  // engine mode: recompute daily after this UTC hour
input string  InpTargetsFile   = "cta_targets.csv";   // targets file in MQL5\Files (file mode)
input string  InpFillsFile     = "cta_fills.csv";     // fills log in MQL5\Files
input string  InpStateFile     = "cta_state.txt";     // last executed rebalance id
input string  InpKillFile      = "cta_halt.txt";      // if present: do nothing
input long    InpMagic         = 20260915;            // magic number
input int     InpTimerSeconds  = 300;                 // check interval
input int     InpRollDays      = 5;                   // roll dated contracts this many days before expiry
input int     InpDeviationPts  = 50;                  // max slippage, points
input double  InpMaxFileAgeH   = 36.0;                // refuse targets older than this (hours, file mode)
input int     InpMaxTargetAgeD = 4;                   // engine mode: stop acting on cached targets older than this (days)
input int     InpMaxDataAgeD   = 5;                   // engine mode: stop acting if the newest PRICE is older than this (days)
input double  InpMaxGrossX     = 6.0;                 // refuse targets with gross notional above this x equity
input int     InpAtrPeriod     = 14;                  // D1 ATR period for the stop rail
input bool    InpDryRun        = false;               // log what would be traded, send nothing

CTrade   trade;
string   g_last_rebalance = "";  // last rebalance id with every root reconciled
string   g_pending        = "";  // rebalance id in progress, if any root is still failing
string   g_done[];               // roots already reconciled for g_pending
datetime g_last_maint     = 0;
datetime g_last_engine    = 0;      // day the engine last ran
CTAMeta   g_meta;
CTATarget g_targets[];
bool      g_have_targets  = false;

//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpDeviationPts);
   trade.SetTypeFilling(ORDER_FILLING_FOK);
   trade.SetAsyncMode(false);
   ReadState();
   PrintFormat("CTA: started, last executed rebalance '%s', pending '%s' (%d roots done), dry-run=%s",
               g_last_rebalance, g_pending, ArraySize(g_done), InpDryRun ? "yes" : "no");
   EventSetTimer(InpTimerSeconds);
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason) { EventKillTimer(); }
void OnTick() {}

//+------------------------------------------------------------------+
//| Deals arrive from the server after OrderSend returns, so the       |
//| fills log written at send time has no price. This records every   |
//| executed deal under our magic, with the server's price, in         |
//| cta_deals.csv: the authoritative record for reconciliation.        |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request,
                        const MqlTradeResult &result)
  {
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
   if(!HistoryDealSelect(trans.deal)) return;
   if(HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != InpMagic) return;
   long   dtype  = HistoryDealGetInteger(trans.deal, DEAL_TYPE);
   long   entry  = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
   string side   = (dtype == DEAL_TYPE_BUY) ? "BUY" : (dtype == DEAL_TYPE_SELL ? "SELL" : "OTHER");
   string ent    = (entry == DEAL_ENTRY_IN) ? "in" : (entry == DEAL_ENTRY_OUT ? "out" : (entry == DEAL_ENTRY_INOUT ? "inout" : "other"));
   int h = FileOpen("cta_deals.csv", FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(h == INVALID_HANDLE) return;
   FileSeek(h, 0, SEEK_END);
   FileWrite(h, TimeToString((datetime)HistoryDealGetInteger(trans.deal, DEAL_TIME), TIME_DATE | TIME_SECONDS),
             HistoryDealGetString(trans.deal, DEAL_SYMBOL), side, ent,
             DoubleToString(HistoryDealGetDouble(trans.deal, DEAL_VOLUME), 2),
             DoubleToString(HistoryDealGetDouble(trans.deal, DEAL_PRICE), 5),
             DoubleToString(HistoryDealGetDouble(trans.deal, DEAL_COMMISSION), 2),
             DoubleToString(HistoryDealGetDouble(trans.deal, DEAL_SWAP), 2),
             DoubleToString(HistoryDealGetDouble(trans.deal, DEAL_PROFIT), 2),
             (string)trans.deal, (string)HistoryDealGetInteger(trans.deal, DEAL_ORDER),
             (string)HistoryDealGetInteger(trans.deal, DEAL_POSITION_ID),
             HistoryDealGetString(trans.deal, DEAL_COMMENT));
   FileClose(h);
  }

//+------------------------------------------------------------------+
void OnTimer()
  {
   if(FileIsExist(InpKillFile)) { Print("CTA: halt file present, idle"); return; }
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) || !MQLInfoInteger(MQL_TRADE_ALLOWED))
     { Print("CTA: trading not allowed in terminal/EA settings"); return; }

   CTAMeta   meta;
   CTATarget targets[];
   if(InpMode == CTA_MODE_FILE)
     {
      if(!CTA_ReadTargets(InpTargetsFile, meta, targets)) return;
      if(!TargetsSane(meta, targets, true)) return;
     }
   else
     {
      if(!EngineTargets()) return;
      meta = g_meta;
      ArrayResize(targets, ArraySize(g_targets));
      for(int i = 0; i < ArraySize(g_targets); i++) targets[i] = g_targets[i];
      if(!TargetsSane(meta, targets, false)) return;
     }

   if(meta.rebalance_id != g_last_rebalance)
     {
      // Retry an incomplete rebalance every tick, but say so only once an
      // hour: a symbol the broker keeps rejecting must not flood the log.
      // Roots already reconciled for this rebalance are not touched again:
      // the engine reruns daily with fresh equity and prices, and without
      // this a single rejected symbol turned the monthly rebalance into
      // daily 0.01-lot top-ups on the rest of the book.
      static datetime last_shout = 0;
      bool shout = (TimeCurrent() - last_shout >= 3600);
      if(g_pending != meta.rebalance_id) { g_pending = meta.rebalance_id; ArrayResize(g_done, 0); }
      if(shout) PrintFormat("CTA: rebalance %s (asof %s, engine %s), %d of %d roots done",
                            meta.rebalance_id, meta.asof, meta.engine, ArraySize(g_done), ArraySize(targets));
      bool all_ok = true;
      for(int i = 0; i < ArraySize(targets); i++)
        {
         if(IsDone(targets[i].root)) continue;
         if(Reconcile(targets[i])) { if(!InpDryRun) MarkDone(targets[i].root); }
         else all_ok = false;
        }
      if(all_ok && !InpDryRun)
        {
         g_last_rebalance = meta.rebalance_id;
         g_pending = ""; ArrayResize(g_done, 0);
         WriteState();
         PrintFormat("CTA: rebalance %s complete", meta.rebalance_id);
        }
      else if(!InpDryRun)
        {
         WriteState();
         if(shout) Print("CTA: rebalance incomplete, will keep retrying the remaining roots");
        }
      if(shout) last_shout = TimeCurrent();
      // fall through: one failing symbol must not block the daily roll and
      // rail maintenance on the rest of the book
     }

   // once a day: roll and trail, whether or not the rebalance is complete
   datetime today = (datetime)(TimeCurrent() / 86400) * 86400;
   if(today == g_last_maint) return;
   bool ok = true;
   for(int i = 0; i < ArraySize(targets); i++)
     {
      if(targets[i].dated && !RollIfNeeded(targets[i])) ok = false;
      if(!TrailRail(targets[i])) ok = false;     // a failed rail update is retried next tick, not next day
     }
   if(ok) g_last_maint = today;
  }

//+------------------------------------------------------------------+
//| The capital the book is sized on. InpCapital fixes it, so the     |
//| allocation does not compound with the account's P&L and the rest  |
//| of the account can hold something else; 0 uses live equity.       |
//+------------------------------------------------------------------+
double Capital()
  {
   if(InpCapital > 0) return InpCapital;
   return AccountInfoDouble(ACCOUNT_EQUITY);
  }

//+------------------------------------------------------------------+
//| Refuse a file that is stale or asks for an implausible book.      |
//+------------------------------------------------------------------+
bool TargetsSane(const CTAMeta &meta, const CTATarget &t[], const bool check_age)
  {
   if(check_age)
     {
      string s = meta.asof;
      StringReplace(s, "T", " ");
      StringReplace(s, "Z", "");
      datetime asof = StringToTime(s);
      double age_h = (double)(TimeGMT() - asof) / 3600.0;
      if(age_h > InpMaxFileAgeH)
        {
         PrintFormat("CTA: targets are %.1f hours old, refusing", age_h);
         return false;
        }
     }
   double gross = 0.0;
   for(int i = 0; i < ArraySize(t); i++)
     {
      if(t[i].side < -1 || t[i].side > 1 || t[i].lots < 0 || !MathIsValidNumber(t[i].lots) || !MathIsValidNumber(t[i].notional))
        { PrintFormat("CTA: target %s has side %d lots %.2f, refusing", t[i].root, t[i].side, t[i].lots); return false; }
      for(int k = 0; k < i; k++) if(t[k].root == t[i].root) { PrintFormat("CTA: duplicate target root %s, refusing", t[i].root); return false; }
      // value the requested lots independently of the file's own notional column
      string sym = t[i].dated ? CTA_ResolveContract(t[i].root, InpRollDays) : t[i].root;
      double lv = (sym == "") ? 0.0 : CTA_LotValueUsd(sym, CTA_SpecCcy(t[i].root));
      double own = (lv > 0 && MathIsValidNumber(lv)) ? t[i].lots * lv : MathAbs(t[i].notional);
      gross += MathMax(MathAbs(t[i].notional), own);
     }
   double eq = Capital();
   if(eq > 0 && gross / eq > InpMaxGrossX)
     {
      PrintFormat("CTA: gross notional %.0f is %.1fx equity, above the %.1fx limit, refusing",
                  gross, gross / eq, InpMaxGrossX);
      return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
//| Engine mode: run the strategy once a day after InpEngineHourUTC   |
//| (and on the first timer tick), keep the targets in memory, and     |
//| write them to cta_targets_engine.csv for comparison with Python.  |
//+------------------------------------------------------------------+
bool EngineTargets()
  {
   MqlDateTime now; TimeToStruct(TimeGMT(), now);
   datetime today = CTA_DayOf(TimeGMT());
   bool due = !g_have_targets || (g_last_engine < today && now.hour >= InpEngineHourUTC);
   if(!due) return g_have_targets;
   string report;
   ulong t0 = GetTickCount64();
   double equity = Capital();
   if(!CTA_ComputeTargets(InpHistoryFile, equity, InpEquityLeg, InpStopAtr, InpRollDays, g_meta, g_targets, report, InpPersistHist))
     {
      Print("CTA: engine failed: ", report);
      // Cached targets are reused for a short while only. After that the EA
      // stops acting rather than trade on stale data; rolls and the rail are
      // skipped too, and the log says so every tick.
      if(g_have_targets && today - g_last_engine > InpMaxTargetAgeD * 86400)
        { PrintFormat("CTA: cached targets are %d days old, refusing to act on them", (int)((today - g_last_engine) / 86400)); g_have_targets = false; }
      return g_have_targets;
     }
   // The engine "succeeding" is not the same as the data being current.
   // CTA_ExtendHistory returns true when it finds nothing to add, so a
   // silent splice failure yields fresh-looking targets computed on frozen
   // prices, and InpMaxTargetAgeD never fires because it measures when the
   // ENGINE last ran, not how old the PRICES are. That happened live: the
   // demo traded on 2026.09.18 prices from 19 to 28 September with no
   // warning. Guard the data itself.
   datetime asof = StringToTime(g_meta.asof);
   int data_age = (int)((today - asof) / 86400);
   if(asof > 0 && data_age > InpMaxDataAgeD)
     {
      PrintFormat("CTA: newest price is %s, %d days old (limit %d). Splice is not "
                  "advancing; refusing to act. Targets NOT updated.",
                  g_meta.asof, data_age, InpMaxDataAgeD);
      g_have_targets = false;
      return false;
     }
   g_have_targets = true; g_last_engine = today;
   CTA_WriteTargets("cta_targets_engine.csv", g_meta, g_targets);
   PrintFormat("CTA: engine ran in %.0fs, asof %s (%d days old), rebalance %s, equity %.0f. splice: %s",
               (GetTickCount64() - t0) / 1000.0, g_meta.asof, data_age, g_meta.rebalance_id, equity, report);
   for(int i = 0; i < ArraySize(g_targets); i++)
      if(g_targets[i].lots > 0)
         PrintFormat("CTA:   %-10s %s %.2f lots  w %.4f", g_targets[i].root, g_targets[i].side > 0 ? "LONG " : "SHORT",
                     g_targets[i].lots, g_targets[i].weight);
   return true;
  }

//+------------------------------------------------------------------+
//| Bring one root to its target. True when done or nothing to do.    |
//+------------------------------------------------------------------+
bool Reconcile(const CTATarget &t)
  {
   string sym = t.dated ? CTA_ResolveContract(t.root, InpRollDays) : t.root;
   if(sym == "")
     {
      PrintFormat("CTA: no contract for %s outside the %d-day roll window", t.root, InpRollDays);
      return false;
     }
   if(!SymbolSelect(sym, true)) { PrintFormat("CTA: cannot select %s", sym); return false; }

   // positions of this root in any OTHER symbol (an expiring contract): close them
   if(t.dated && !CloseOtherContracts(t.root, sym)) return false;

   double target  = t.side * CTA_NormalizeLots(sym, t.lots);
   double current = CTA_NetLots(sym, InpMagic);
   double vstep   = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
   if(MathAbs(target - current) <= vstep * 0.5 + 1e-9) return true;

   // sign flip or flatten: close everything first
   if(current != 0.0 && (target == 0.0 || (target > 0) != (current > 0)))
     {
      if(!CloseAll(sym, "flip")) return false;
      current = 0.0;
     }
   double delta = target - current;
   if(MathAbs(delta) <= vstep * 0.5 + 1e-9) return true;

   if(current != 0.0 && MathAbs(target) < MathAbs(current))
      return ReduceBy(sym, MathAbs(delta), "reduce");
   return Open(sym, delta > 0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL, MathAbs(delta), t.stop_atr, "rebalance");
  }

//+------------------------------------------------------------------+
bool Open(const string sym, const ENUM_ORDER_TYPE type, const double lots, const double stop_atr,
          const string note)
  {
   double v = CTA_NormalizeLots(sym, lots);
   if(v == 0.0) return true;
   double sl = 0.0;
   if(stop_atr > 0)
     {
      double atr = Atr(sym);
      // No ATR means no rail. A rail was asked for, so do not enter
      // unprotected; fail the order and let the next tick retry.
      if(atr <= 0) { PrintFormat("CTA: no ATR for %s, not opening without a rail", sym); return false; }
      double px = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(sym, SYMBOL_ASK)
                                           : SymbolInfoDouble(sym, SYMBOL_BID);
      sl = (type == ORDER_TYPE_BUY) ? px - stop_atr * atr : px + stop_atr * atr;
      sl = NormalizeDouble(sl, (int)SymbolInfoInteger(sym, SYMBOL_DIGITS));
     }
   if(InpDryRun)
     {
      PrintFormat("CTA[dry]: %s %s %.2f sl %.5f (%s)", type == ORDER_TYPE_BUY ? "BUY" : "SELL", sym, v, sl, note);
      return true;
     }
   bool ok = (type == ORDER_TYPE_BUY) ? trade.Buy(v, sym, 0.0, sl, 0.0, note)
                                      : trade.Sell(v, sym, 0.0, sl, 0.0, note);
   uint rc = trade.ResultRetcode();
   CTA_LogFill(InpFillsFile, sym, type == ORDER_TYPE_BUY ? "BUY" : "SELL", v,
               trade.ResultPrice(), trade.ResultDeal(), rc, note);
   if(!ok || !RetcodeOk(rc))
     {
      PrintFormat("CTA: %s %s %.2f failed: %u %s", type == ORDER_TYPE_BUY ? "BUY" : "SELL", sym, v, rc,
                  trade.ResultRetcodeDescription());
      return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
bool CloseAll(const string sym, const string note)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != sym) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      double v = PositionGetDouble(POSITION_VOLUME);
      if(InpDryRun) { PrintFormat("CTA[dry]: CLOSE %s #%I64u %.2f (%s)", sym, ticket, v, note); continue; }
      bool ok = trade.PositionClose(ticket);
      uint rc = trade.ResultRetcode();
      CTA_LogFill(InpFillsFile, sym, "CLOSE", v, trade.ResultPrice(), trade.ResultDeal(), rc, note);
      // PositionClose returns true when the request was SENT; a server
      // rejection comes back as true with a failing retcode. Check both.
      if(!ok || !RetcodeOk(rc))
        { PrintFormat("CTA: close %s #%I64u failed: %u %s", sym, ticket, rc, trade.ResultRetcodeDescription()); return false; }
     }
   if(!InpDryRun && MathAbs(CTA_NetLots(sym, InpMagic)) > 0.0)
     { PrintFormat("CTA: close %s reported done but %.2f lots remain", sym, CTA_NetLots(sym, InpMagic)); return false; }
   return true;
  }

bool RetcodeOk(const uint rc) { return rc == TRADE_RETCODE_DONE || rc == TRADE_RETCODE_PLACED || rc == TRADE_RETCODE_DONE_PARTIAL; }

//+------------------------------------------------------------------+
//| Reduce the net position in sym by `amount` lots, closing whole    |
//| positions first and one partial at the end.                       |
//+------------------------------------------------------------------+
bool ReduceBy(const string sym, double amount, const string note)
  {
   double vstep = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
   for(int i = PositionsTotal() - 1; i >= 0 && amount > vstep * 0.5; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != sym) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      double v = PositionGetDouble(POSITION_VOLUME);
      double c = MathMin(v, CTA_NormalizeLots(sym, amount));
      if(c == 0.0) return true;                  // residual below the minimum lot: nothing tradeable is left
      bool ok;
      if(InpDryRun) { PrintFormat("CTA[dry]: REDUCE %s #%I64u by %.2f (%s)", sym, ticket, c, note); amount -= c; continue; }
      ok = (c >= v) ? trade.PositionClose(ticket) : trade.PositionClosePartial(ticket, c);
      uint rc = trade.ResultRetcode();
      CTA_LogFill(InpFillsFile, sym, "REDUCE", c, trade.ResultPrice(), trade.ResultDeal(), rc, note);
      if(!ok || !RetcodeOk(rc)) { PrintFormat("CTA: reduce %s #%I64u failed: %u %s", sym, ticket, rc, trade.ResultRetcodeDescription()); return false; }
      amount -= c;
     }
   return amount <= vstep * 0.5 + 1e-9;
  }

//+------------------------------------------------------------------+
//| Close positions of `root` held in any contract other than `keep`. |
//+------------------------------------------------------------------+
bool CloseOtherContracts(const string root, const string keep)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      string s = PositionGetString(POSITION_SYMBOL);
      if(s == keep || !CTA_SymbolOfRoot(s, root, true)) continue;
      if(!CloseAll(s, "roll-out")) return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
//| Between rebalances: if the held contract is inside the roll       |
//| window, move the same net lots to the next one.                   |
//+------------------------------------------------------------------+
bool RollIfNeeded(const CTATarget &t)
  {
   string next = CTA_ResolveContract(t.root, InpRollDays);
   if(next == "") { PrintFormat("CTA: no roll target for %s", t.root); return false; }
   // Open the replacement BEFORE closing the expiring contract. The old
   // order (close, then open) lost the exposure whenever the open was
   // rejected: the next day found nothing to roll and reported success.
   // Opening first means a rejected open leaves the old position in place
   // to retry tomorrow, at the cost of a few minutes of double exposure.
   double moved = 0.0; string olds[]; int n_old = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      string s = PositionGetString(POSITION_SYMBOL);
      if(s == next || !CTA_SymbolOfRoot(s, t.root, true)) continue;
      bool seen = false;
      for(int k = 0; k < n_old; k++) if(olds[k] == s) seen = true;
      if(seen) continue;
      ArrayResize(olds, n_old + 1); olds[n_old++] = s;
      moved += CTA_NetLots(s, InpMagic);
     }
   if(n_old == 0) return true;
   double already = CTA_NetLots(next, InpMagic);      // a previous attempt may have opened but failed to close
   double need = moved - already;
   PrintFormat("CTA: rolling %s -> %s, %.2f lots (%.2f already in place)", olds[0], next, moved, already);
   SymbolSelect(next, true);
   double vstep = SymbolInfoDouble(next, SYMBOL_VOLUME_STEP);
   if(MathAbs(need) >= vstep * 0.5)
      if(!Open(next, need > 0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL, MathAbs(need), t.stop_atr, "roll-in")) return false;
   for(int k = 0; k < n_old; k++)
      if(!CloseAll(olds[k], "roll-out")) return false;
   return true;
  }

//+------------------------------------------------------------------+
//| Ratchet the stop rail: only ever tightens.                        |
//+------------------------------------------------------------------+
bool TrailRail(const CTATarget &t)
  {
   bool all_ok = true;
   if(t.stop_atr <= 0) return true;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      string s = PositionGetString(POSITION_SYMBOL);
      if(!CTA_SymbolOfRoot(s, t.root, t.dated)) continue;
      double atr = Atr(s);
      if(atr <= 0) { PrintFormat("CTA: no ATR for %s, rail not updated", s); all_ok = false; continue; }
      int digits = (int)SymbolInfoInteger(s, SYMBOL_DIGITS);
      double cur = PositionGetDouble(POSITION_SL);
      double sl;
      if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
        {
         sl = NormalizeDouble(SymbolInfoDouble(s, SYMBOL_BID) - t.stop_atr * atr, digits);
         if(cur != 0.0 && sl <= cur) continue;
        }
      else
        {
         sl = NormalizeDouble(SymbolInfoDouble(s, SYMBOL_ASK) + t.stop_atr * atr, digits);
         if(cur != 0.0 && sl >= cur) continue;
        }
      if(InpDryRun) { PrintFormat("CTA[dry]: SL %s #%I64u -> %.5f", s, ticket, sl); continue; }
      bool ok = trade.PositionModify(ticket, sl, PositionGetDouble(POSITION_TP));
      if(!ok || !RetcodeOk(trade.ResultRetcode()))
        { PrintFormat("CTA: modify SL %s #%I64u failed: %u %s", s, ticket, trade.ResultRetcode(), trade.ResultRetcodeDescription()); all_ok = false; }
     }
   return all_ok;
  }

//+------------------------------------------------------------------+
//| D1 ATR, last completed bar. 0 on failure.                         |
//+------------------------------------------------------------------+
double Atr(const string sym)
  {
   double a = AtrOf(sym);
   if(a > 0) return a;
   // A newly listed dated contract can have fewer bars than the ATR period.
   // The rolling CFD of the same root (GOLDZ6 -> GOLD) trades at the same
   // price scale and has years of history; use its ATR for the rail.
   if(SymbolInfoInteger(sym, SYMBOL_EXPIRATION_TIME) > 0 && StringLen(sym) > 2)
     {
      string root = StringSubstr(sym, 0, StringLen(sym) - 2);
      if(SymbolSelect(root, true)) { a = AtrOf(root); if(a > 0) PrintFormat("CTA: ATR for %s taken from %s", sym, root); }
     }
   return a;
  }

double AtrOf(const string sym)
  {
   int h = iATR(sym, PERIOD_D1, InpAtrPeriod);
   if(h == INVALID_HANDLE) return 0.0;
   double buf[];
   if(CopyBuffer(h, 0, 1, 1, buf) != 1) { IndicatorRelease(h); return 0.0; }
   IndicatorRelease(h);
   return (MathIsValidNumber(buf[0]) && buf[0] > 0) ? buf[0] : 0.0;
  }

//+------------------------------------------------------------------+
//| State file. Line 1: last complete rebalance id. Line 2, present   |
//| only while a rebalance is partly done: pending id and the roots    |
//| already reconciled, pipe-separated. A one-line file from the       |
//| previous format reads as "complete, nothing pending".              |
//+------------------------------------------------------------------+
void ReadState()
  {
   g_last_rebalance = ""; g_pending = ""; ArrayResize(g_done, 0);
   if(!FileIsExist(InpStateFile)) return;
   int h = FileOpen(InpStateFile, FILE_READ | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE) return;
   g_last_rebalance = FileReadString(h);
   if(!FileIsEnding(h))
     {
      string line = FileReadString(h);
      string parts[];
      int k = StringSplit(line, '|', parts);
      if(k >= 1 && parts[0] != "")
        {
         g_pending = parts[0];
         for(int i = 1; i < k; i++)
            if(parts[i] != "") MarkDone(parts[i]);
        }
     }
   FileClose(h);
  }

void WriteState()
  {
   int h = FileOpen(InpStateFile, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE) { PrintFormat("CTA: cannot write state (%d)", GetLastError()); return; }
   FileWrite(h, g_last_rebalance);
   if(g_pending != "")
     {
      string line = g_pending;
      for(int i = 0; i < ArraySize(g_done); i++) line += "|" + g_done[i];
      FileWrite(h, line);
     }
   FileClose(h);
  }

bool IsDone(const string root)
  {
   for(int i = 0; i < ArraySize(g_done); i++) if(g_done[i] == root) return true;
   return false;
  }

void MarkDone(const string root)
  {
   if(IsDone(root)) return;
   int k = ArraySize(g_done);
   ArrayResize(g_done, k + 1);
   g_done[k] = root;
  }
//+------------------------------------------------------------------+
