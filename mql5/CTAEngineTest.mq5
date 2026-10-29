//+------------------------------------------------------------------+
//| CTAEngineTest.mq5                                                 |
//| Script: run CTAEngine.mqh on the exported research price matrix   |
//| and write its outputs for comparison with the Python reference.   |
//| Input  MQL5\Files\cta_port_prices.csv  (live/port_check.py export) |
//| Output MQL5\Files\cta_port_mql5.csv                                |
//+------------------------------------------------------------------+
#property script_show_inputs
#property strict
#include "CTAEngine.mqh"

input string InpPrices = "cta_port_prices.csv";
input string InpOut    = "cta_port_mql5.csv";

void OnStart()
  {
   int h = FileOpen(InpPrices, FILE_READ | FILE_CSV | FILE_ANSI, ',');
   if(h == INVALID_HANDLE) { PrintFormat("cannot open %s (%d)", InpPrices, GetLastError()); return; }
   // header: date,sym1,...,symm
   string hdr[]; int m = 0;
   FileReadString(h);                          // "date"
   while(!FileIsLineEnding(h) && !FileIsEnding(h))
     {
      string s = FileReadString(h);
      ArrayResize(hdr, m + 1); hdr[m++] = s;
     }
   PrintFormat("%d instruments: %s ...", m, hdr[0]);
   // buckets in specs order: spx nasdaq nikkei | gold silver copper wti brent | aud gbp jpy | us10y gilt
   int bucket[]; ArrayResize(bucket, m);
   for(int j = 0; j < m; j++)
     {
      string s = hdr[j];
      if(s == "spx" || s == "nasdaq" || s == "nikkei") bucket[j] = 0;
      else if(s == "gold" || s == "silver" || s == "copper" || s == "wti" || s == "brent") bucket[j] = 1;
      else if(s == "aud" || s == "gbp" || s == "jpy") bucket[j] = 2;
      else bucket[j] = 3;
     }
   double px[]; datetime dates[]; int n = 0;
   while(!FileIsEnding(h))
     {
      string d = FileReadString(h);
      if(d == "") break;
      ArrayResize(dates, n + 1); ArrayResize(px, (n + 1) * m);
      dates[n] = StringToTime(d);
      for(int j = 0; j < m; j++)
        {
         string v = FileReadString(h);
         px[n * m + j] = (v == "") ? CTA_NAN() : StringToDouble(v);
        }
      n++;
     }
   FileClose(h);
   PrintFormat("%d days, %s -> %s", n, TimeToString(dates[0], TIME_DATE), TimeToString(dates[n - 1], TIME_DATE));

   CTAParams p; CTA_DefaultParams(p);
   bool is_rebal[]; CTA_MonthlyRebal(dates, n, is_rebal);
   double w[], lev[], mult[];
   ulong t0 = GetTickCount64();
   CTA_BuildTargets(px, n, m, dates, bucket, 4, is_rebal, p, w, lev, mult);
   PrintFormat("engine run in %.1fs", (GetTickCount64() - t0) / 1000.0);

   int o = FileOpen(InpOut, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(o == INVALID_HANDLE) { PrintFormat("cannot write %s (%d)", InpOut, GetLastError()); return; }
   string line = "date,leverage,mult";
   for(int j = 0; j < m; j++) line += "," + hdr[j];
   FileWrite(o, line);
   for(int t = 0; t < n; t++)
     {
      line = TimeToString(dates[t], TIME_DATE) + "," + DoubleToString(lev[t], 12) + "," + DoubleToString(mult[t], 12);
      for(int j = 0; j < m; j++) line += "," + DoubleToString(w[t * m + j], 12);
      FileWrite(o, line);
     }
   FileClose(o);
   PrintFormat("wrote %s", InpOut);
  }
