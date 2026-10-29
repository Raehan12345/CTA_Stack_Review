//+------------------------------------------------------------------+
//| CTATargets.mqh                                                    |
//| Targets file parsing, contract resolution, position netting and   |
//| fill logging for CTAExecutor.mq5. No strategy logic lives here.   |
//+------------------------------------------------------------------+
#property strict

struct CTATarget
  {
   string            root;        // broker symbol or dated root, e.g. SPX500 / TNOTE10YR
   bool              dated;       // resolve root -> nearest listed contract
   int               side;        // -1, 0, +1
   double            lots;        // absolute
   double            weight;
   double            notional;
   double            stop_atr;    // 0 = no rail
  };

struct CTAMeta
  {
   string            asof;
   string            rebalance_id;
   double            equity;
   int               n;
   string            engine;
  };

//+------------------------------------------------------------------+
//| Read cta_targets.csv. Returns false on any structural problem.    |
//+------------------------------------------------------------------+
bool CTA_ReadTargets(const string fname, CTAMeta &meta, CTATarget &out[])
  {
   int h = FileOpen(fname, FILE_READ | FILE_CSV | FILE_ANSI, ',');
   if(h == INVALID_HANDLE)
     {
      PrintFormat("CTA: cannot open %s (%d)", fname, GetLastError());
      return false;
     }
   // line 1: META,asof,rebalance_id,equity,n,engine
   string tag = FileReadString(h);
   if(tag != "META")
     {
      Print("CTA: targets file does not start with META");
      FileClose(h);
      return false;
     }
   meta.asof         = FileReadString(h);
   meta.rebalance_id = FileReadString(h);
   meta.equity       = StringToDouble(FileReadString(h));
   meta.n            = (int)StringToInteger(FileReadString(h));
   meta.engine       = FileReadString(h);
   // line 2: header, 7 fields
   for(int i = 0; i < 7; i++) FileReadString(h);
   ArrayResize(out, 0);
   while(!FileIsEnding(h))
     {
      string root = FileReadString(h);
      if(root == "" || FileIsEnding(h)) break;
      CTATarget t;
      t.root     = root;
      t.dated    = (FileReadString(h) == "dated");
      t.side     = (int)StringToInteger(FileReadString(h));
      t.lots     = StringToDouble(FileReadString(h));
      t.weight   = StringToDouble(FileReadString(h));
      t.notional = StringToDouble(FileReadString(h));
      t.stop_atr = StringToDouble(FileReadString(h));
      int k = ArraySize(out);
      ArrayResize(out, k + 1);
      out[k] = t;
     }
   FileClose(h);
   if(ArraySize(out) != meta.n)
     {
      PrintFormat("CTA: META says %d rows, read %d", meta.n, ArraySize(out));
      return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
//| Nearest listed contract for a dated root whose expiry is more     |
//| than roll_days away. "" if none.                                  |
//+------------------------------------------------------------------+
string CTA_ResolveContract(const string root, const int roll_days)
  {
   string   best = "";
   datetime best_exp = 0;
   datetime now = TimeCurrent();
   int total = SymbolsTotal(false);
   int rl = StringLen(root);
   for(int i = 0; i < total; i++)
     {
      string s = SymbolName(i, false);
      if(StringSubstr(s, 0, rl) != root) continue;
      // the rolling CFD shares the root (e.g. GOLD vs GOLDZ6): require an expiry
      datetime exp = (datetime)SymbolInfoInteger(s, SYMBOL_EXPIRATION_TIME);
      if(exp == 0) continue;
      if(exp - now < roll_days * 86400) continue;
      if(best == "" || exp < best_exp) { best = s; best_exp = exp; }
     }
   return best;
  }

//+------------------------------------------------------------------+
//| Does symbol s belong to root (exact CFD name, or dated contract)? |
//+------------------------------------------------------------------+
bool CTA_SymbolOfRoot(const string s, const string root, const bool dated)
  {
   if(!dated) return s == root;
   int rl = StringLen(root);
   if(StringSubstr(s, 0, rl) != root) return false;
   return SymbolInfoInteger(s, SYMBOL_EXPIRATION_TIME) > 0;
  }

//+------------------------------------------------------------------+
//| Net signed lots held in symbol under magic (hedging account).     |
//+------------------------------------------------------------------+
double CTA_NetLots(const string symbol, const long magic)
  {
   double net = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != magic) continue;
      double v = PositionGetDouble(POSITION_VOLUME);
      net += (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? v : -v;
     }
   return net;
  }

//+------------------------------------------------------------------+
//| Append one line to the fills log.                                 |
//+------------------------------------------------------------------+
void CTA_LogFill(const string fname, const string symbol, const string action, const double lots,
                 const double price, const ulong deal, const uint retcode, const string note)
  {
   int h = FileOpen(fname, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(h == INVALID_HANDLE) { PrintFormat("CTA: cannot open fills log (%d)", GetLastError()); return; }
   FileSeek(h, 0, SEEK_END);
   FileWrite(h, TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS), symbol, action,
             DoubleToString(lots, 2), DoubleToString(price, 5), (string)deal, (string)retcode, note);
   FileClose(h);
  }

//+------------------------------------------------------------------+
//| Round lots to the symbol's step, floor to zero below the minimum. |
//+------------------------------------------------------------------+
double CTA_NormalizeLots(const string symbol, const double lots)
  {
   double vmin  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double vstep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   double v = MathFloor(MathAbs(lots) / vstep + 1e-9) * vstep;
   if(v < vmin) return 0.0;
   return v;
  }
