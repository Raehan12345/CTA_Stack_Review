//+------------------------------------------------------------------+
//| CTAEngine.mqh                                                     |
//| The CTA trend sleeve, function for function against               |
//| cta/engine.py::build_trend_nav   with           |
//|   hrp=True, regime_gate=True, regime_mult_mode="learned",          |
//|   regime_features=(vol,corr,er,disp,agree), regime_auto_k=(2,10),  |
//|   regime_kmeans="portable", hrp_lag=1, rebalance_freq="M".         |
//|                                                                    |
//| Data model: a daily close matrix px[t*m + j] over a common calendar |
//| (t = day, j = instrument), NaN where an instrument has no close.    |
//| pandas semantics are reproduced deliberately: any NaN inside a     |
//| rolling window makes that window NaN; cross-sectional means skip   |
//| NaN; sample statistics use ddof=1.                                 |
//|                                                                    |
//| Output: target weights (fraction of NAV, leverage-scaled) per day. |
//+------------------------------------------------------------------+
#property strict

double CTA_NAN() { return MathSqrt(-1.0); }
bool   CTA_IsNaN(const double x) { return !MathIsValidNumber(x); }

//============================================================ parameters
struct CTAParams
  {
   int    short_lookback;     // 40
   int    long_lookback;      // 252
   int    vol_window;         // 60
   double inst_target_vol;    // 0.20
   double max_inst_weight;    // 3.0
   double base_target_vol;    // 0.13
   double max_leverage;       // 4.0
   int    hrp_window;         // 252
   int    hrp_lag;            // 1
   int    er_window;          // 40
   int    corr_window;        // 60
   int    regime_min_train;   // 504
   int    regime_refit_every; // 13
   int    regime_hysteresis;  // 2
   int    regime_fwd_window;  // 21
   int    regime_k_min;       // 2
   int    regime_k_max;       // 10
   int    kmeans_n_init;      // 10
   int    kmeans_max_iter;    // 300
   int    kmeans_seed;        // 0
   double ladder_top;         // 1.0
   double ladder_bottom;      // 0.5
  };

void CTA_DefaultParams(CTAParams &p)
  {
   p.short_lookback = 40;  p.long_lookback = 252; p.vol_window = 60;
   p.inst_target_vol = 0.20; p.max_inst_weight = 3.0; p.base_target_vol = 0.13; p.max_leverage = 4.0;
   p.hrp_window = 252; p.hrp_lag = 1; p.er_window = 40; p.corr_window = 60;
   p.regime_min_train = 504; p.regime_refit_every = 13; p.regime_hysteresis = 2; p.regime_fwd_window = 21;
   p.regime_k_min = 2; p.regime_k_max = 10; p.kmeans_n_init = 10; p.kmeans_max_iter = 300; p.kmeans_seed = 0;
   p.ladder_top = 1.0; p.ladder_bottom = 0.5;
  }

//============================================================ rolling helpers (column j of an n*m matrix)
// Rolling sample std over `w` rows ending at t (inclusive); NaN unless all w values are valid.
double CTA_RollStd(const double &x[], const int n, const int m, const int j, const int t, const int w)
  {
   if(t - w + 1 < 0) return CTA_NAN();
   double s = 0, s2 = 0;
   for(int i = t - w + 1; i <= t; i++)
     {
      double v = x[i * m + j];
      if(CTA_IsNaN(v)) return CTA_NAN();
      s += v; s2 += v * v;
     }
   double mean = s / w;
   double var = (s2 - w * mean * mean) / (w - 1);
   if(var < 0) var = 0;
   return MathSqrt(var);
  }

double CTA_RollMean(const double &x[], const int n, const int m, const int j, const int t, const int w)
  {
   if(t - w + 1 < 0) return CTA_NAN();
   double s = 0;
   for(int i = t - w + 1; i <= t; i++)
     {
      double v = x[i * m + j];
      if(CTA_IsNaN(v)) return CTA_NAN();
      s += v;
     }
   return s / w;
  }

double CTA_RollVar(const double &x[], const int n, const int m, const int j, const int t, const int w)
  {
   double sd = CTA_RollStd(x, n, m, j, t, w);
   return CTA_IsNaN(sd) ? sd : sd * sd;
  }

double CTA_Sign(const double v)
  {
   if(CTA_IsNaN(v)) return CTA_NAN();
   return v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0);
  }

//============================================================ 1. returns and score
// r[t,j] = px[t,j]/px[t-k,j] - 1  (NaN propagates, as pandas 3 pct_change)
void CTA_PctChange(const double &px[], const int n, const int m, const int k, double &r[])
  {
   ArrayResize(r, n * m);
   for(int t = 0; t < n; t++)
      for(int j = 0; j < m; j++)
        {
         if(t < k) { r[t * m + j] = CTA_NAN(); continue; }
         double a = px[t * m + j], b = px[(t - k) * m + j];
         r[t * m + j] = (CTA_IsNaN(a) || CTA_IsNaN(b)) ? CTA_NAN() : a / b - 1.0;
        }
  }

// score[t,j] = (0.5*sign(r_short) + 0.5*sign(r_long)) shifted one day
void CTA_TsmomScore(const double &px[], const int n, const int m, const CTAParams &p, double &score[])
  {
   double rs[], rl[];
   CTA_PctChange(px, n, m, p.short_lookback, rs);
   CTA_PctChange(px, n, m, p.long_lookback, rl);
   ArrayResize(score, n * m);
   for(int t = 0; t < n; t++)
      for(int j = 0; j < m; j++)
        {
         if(t == 0) { score[j] = CTA_NAN(); continue; }
         double a = CTA_Sign(rs[(t - 1) * m + j]), b = CTA_Sign(rl[(t - 1) * m + j]);
         score[t * m + j] = (CTA_IsNaN(a) || CTA_IsNaN(b)) ? CTA_NAN() : 0.5 * a + 0.5 * b;
        }
  }

// Kaufman efficiency ratio per instrument, fillna(0) then shifted one day.
void CTA_EfficiencyRatio(const double &px[], const int n, const int m, const int w, double &er[])
  {
   double d[]; ArrayResize(d, n * m);
   for(int t = 0; t < n; t++)
      for(int j = 0; j < m; j++)
        {
         if(t == 0) { d[j] = CTA_NAN(); continue; }
         double a = px[t * m + j], b = px[(t - 1) * m + j];
         d[t * m + j] = (CTA_IsNaN(a) || CTA_IsNaN(b)) ? CTA_NAN() : MathAbs(a - b);
        }
   double raw[]; ArrayResize(raw, n * m);
   for(int t = 0; t < n; t++)
      for(int j = 0; j < m; j++)
        {
         double v = CTA_NAN();
         if(t >= w)
           {
            double a = px[t * m + j], b = px[(t - w) * m + j];
            double tot = CTA_RollMean(d, n, m, j, t, w);
            if(!CTA_IsNaN(a) && !CTA_IsNaN(b) && !CTA_IsNaN(tot))
              {
               tot *= w;
               // tot is the sum of |daily moves| over the window, so tot == 0
               // implies a == b and pandas gives 0/0 = NaN, filled to 0 below.
               // MQL5 treats a zero divide as a fatal runtime error, so guard it.
               v = (tot > 0) ? MathAbs(a - b) / tot : CTA_NAN();
              }
           }
         if(CTA_IsNaN(v)) v = 0.0;             // .fillna(0.0)
         raw[t * m + j] = v;
        }
   ArrayResize(er, n * m);
   for(int t = 0; t < n; t++)
      for(int j = 0; j < m; j++)
         er[t * m + j] = (t == 0) ? CTA_NAN() : raw[(t - 1) * m + j];
  }

//============================================================ 2. HRP (pypfopt HRPOpt, single linkage)
// Single-linkage dendrogram pre-order on distance sqrt((1-corr)/2), then
// recursive bisection with inverse-variance cluster weights. Weights are
// returned times the bucket size (scale relative to equal weight).
void CTA_HrpWeights(const double &cov[], const double &corr[], const int k, double &w_out[])
  {
   // --- distance matrix
   double dist[]; ArrayResize(dist, k * k);
   for(int a = 0; a < k; a++)
      for(int b = 0; b < k; b++)
        {
         double v = (1.0 - corr[a * k + b]) / 2.0;
         if(v < 0) v = 0; if(v > 1) v = 1;
         dist[a * k + b] = MathSqrt(v);
        }
   // --- single linkage: clusters as member lists; ids 0..k-1 leaves, k.. merges
   int    n_nodes = 2 * k - 1;
   int    left[], right[];  ArrayResize(left, n_nodes); ArrayResize(right, n_nodes);
   for(int i = 0; i < n_nodes; i++) { left[i] = -1; right[i] = -1; }
   int    active[]; ArrayResize(active, k);
   int    members[]; ArrayResize(members, n_nodes * k);       // members[node*k + i], -1 terminated
   int    msize[];   ArrayResize(msize, n_nodes);
   for(int i = 0; i < k; i++) { active[i] = i; members[i * k] = i; msize[i] = 1; }
   int n_active = k;
   for(int step = 0; step < k - 1; step++)
     {
      double best = 1e300; int bi = -1, bj = -1;
      for(int a = 0; a < n_active; a++)
         for(int b = a + 1; b < n_active; b++)
           {
            int ca = active[a], cb = active[b];
            double dmin = 1e300;
            for(int x = 0; x < msize[ca]; x++)
               for(int y = 0; y < msize[cb]; y++)
                 {
                  double dv = dist[members[ca * k + x] * k + members[cb * k + y]];
                  if(dv < dmin) dmin = dv;
                 }
            if(dmin < best) { best = dmin; bi = a; bj = b; }
           }
      int ca = active[bi], cb = active[bj];
      int node = k + step;
      // scipy orders each merge row with the smaller cluster id first
      left[node]  = MathMin(ca, cb);
      right[node] = MathMax(ca, cb);
      msize[node] = msize[ca] + msize[cb];
      for(int x = 0; x < msize[ca]; x++) members[node * k + x] = members[ca * k + x];
      for(int y = 0; y < msize[cb]; y++) members[node * k + msize[ca] + y] = members[cb * k + y];
      active[bi] = node;
      for(int a = bj; a < n_active - 1; a++) active[a] = active[a + 1];
      n_active--;
     }
   // --- pre-order traversal of the root
   int order[]; ArrayResize(order, k); int pos = 0;
   int stack[]; ArrayResize(stack, n_nodes); int sp = 0;
   stack[sp++] = n_nodes - 1;
   while(sp > 0)
     {
      int node = stack[--sp];
      if(node < k) { order[pos++] = node; continue; }
      stack[sp++] = right[node];
      stack[sp++] = left[node];
     }
   // --- recursive bisection on the ordered list
   double w[]; ArrayResize(w, k); for(int i = 0; i < k; i++) w[i] = 1.0;
   // queue of (start, end) index ranges into `order`
   int qs[], qe[]; ArrayResize(qs, 4 * k); ArrayResize(qe, 4 * k);
   int qn = 1; qs[0] = 0; qe[0] = k;
   while(qn > 0)
     {
      int ns2 = 0; int s2[], e2[]; ArrayResize(s2, 4 * k); ArrayResize(e2, 4 * k);
      for(int q = 0; q < qn; q++)
        {
         int len = qe[q] - qs[q];
         if(len <= 1) continue;
         int mid = qs[q] + len / 2;
         s2[ns2] = qs[q]; e2[ns2] = mid; ns2++;
         s2[ns2] = mid;   e2[ns2] = qe[q]; ns2++;
        }
      for(int q = 0; q + 1 < ns2; q += 2)
        {
         double v1 = CTA_ClusterVar(cov, k, order, s2[q], e2[q]);
         double v2 = CTA_ClusterVar(cov, k, order, s2[q + 1], e2[q + 1]);
         double alpha = 1.0 - v1 / (v1 + v2);
         for(int i = s2[q]; i < e2[q]; i++)       w[order[i]] *= alpha;
         for(int i = s2[q + 1]; i < e2[q + 1]; i++) w[order[i]] *= (1.0 - alpha);
        }
      qn = ns2;
      for(int q = 0; q < qn; q++) { qs[q] = s2[q]; qe[q] = e2[q]; }
     }
   ArrayResize(w_out, k);
   for(int i = 0; i < k; i++) w_out[i] = w[i] * k;
  }

double CTA_ClusterVar(const double &cov[], const int k, const int &order[], const int s, const int e)
  {
   int len = e - s;
   double iv[]; ArrayResize(iv, len); double tot = 0;
   for(int i = 0; i < len; i++) { iv[i] = 1.0 / cov[order[s + i] * k + order[s + i]]; tot += iv[i]; }
   for(int i = 0; i < len; i++) iv[i] /= tot;
   double v = 0;
   for(int a = 0; a < len; a++)
      for(int b = 0; b < len; b++)
         v += iv[a] * cov[order[s + a] * k + order[s + b]] * iv[b];
   return v;
  }

// HRP scale per instrument per day: recomputed at rebalance days from the
// trailing hrp_window rows ending hrp_lag days before, carried forward.
void CTA_HrpScale(const double &ret[], const int n, const int m, const int &bucket[], const int n_buckets,
                  const bool &is_rebal[], const CTAParams &p, double &scale[])
  {
   ArrayResize(scale, n * m);
   for(int i = 0; i < n * m; i++) scale[i] = 1.0;
   for(int b = 0; b < n_buckets; b++)
     {
      int idx[]; int k = 0;
      for(int j = 0; j < m; j++) if(bucket[j] == b) { ArrayResize(idx, k + 1); idx[k++] = j; }
      if(k < 2) continue;
      double current[]; ArrayResize(current, k); for(int i = 0; i < k; i++) current[i] = 1.0;
      for(int t = 0; t < n; t++)
        {
         if(is_rebal[t])
           {
            int end = t - p.hrp_lag;                 // inclusive last row of the window
            int start = end - p.hrp_window + 1;
            if(start < 0) start = 0;
            int len = end - start + 1;
            if(len >= p.hrp_window / 2)
              {
               // window must have no NaN and positive std in every column
               bool ok = true;
               double mean[]; ArrayResize(mean, k);
               for(int a = 0; a < k && ok; a++)
                 {
                  double s = 0;
                  for(int r = start; r <= end; r++)
                    {
                     double v = ret[r * m + idx[a]];
                     if(CTA_IsNaN(v)) { ok = false; break; }
                     s += v;
                    }
                  mean[a] = s / len;
                 }
               if(ok)
                 {
                  double cov[]; ArrayResize(cov, k * k);
                  for(int a = 0; a < k; a++)
                     for(int c = a; c < k; c++)
                       {
                        double s = 0;
                        for(int r = start; r <= end; r++)
                           s += (ret[r * m + idx[a]] - mean[a]) * (ret[r * m + idx[c]] - mean[c]);
                        cov[a * k + c] = s / (len - 1); cov[c * k + a] = cov[a * k + c];
                       }
                  for(int a = 0; a < k; a++) if(cov[a * k + a] <= 0) ok = false;
                  if(ok)
                    {
                     double corr[]; ArrayResize(corr, k * k);
                     for(int a = 0; a < k; a++)
                        for(int c = 0; c < k; c++)
                           corr[a * k + c] = cov[a * k + c] / MathSqrt(cov[a * k + a] * cov[c * k + c]);
                     double w[];
                     CTA_HrpWeights(cov, corr, k, w);
                     for(int a = 0; a < k; a++) current[a] = w[a];
                    }
                 }
              }
           }
         for(int a = 0; a < k; a++) scale[t * m + idx[a]] = current[a];
        }
     }
  }

//============================================================ 3. regime features (vol, corr, er, disp, agree), shifted one day
void CTA_RegimeFeatures(const double &px[], const int n, const int m, const CTAParams &p, double &f[])
  {
   const int F = 5;
   double ret[]; CTA_PctChange(px, n, m, 1, ret);
   // equal-weighted book return (mean over valid instruments), cross-sectional std (ddof=1)
   double eq[], xs[]; ArrayResize(eq, n); ArrayResize(xs, n);
   for(int t = 0; t < n; t++)
     {
      double s = 0, s2 = 0; int c = 0;
      for(int j = 0; j < m; j++) { double v = ret[t * m + j]; if(!CTA_IsNaN(v)) { s += v; s2 += v * v; c++; } }
      eq[t] = (c > 0) ? s / c : CTA_NAN();
      if(c > 1) { double mu = s / c; double var = (s2 - c * mu * mu) / (c - 1); xs[t] = MathSqrt(var < 0 ? 0 : var); }
      else xs[t] = CTA_NAN();
     }
   double er[]; CTA_EfficiencyRatio(px, n, m, p.er_window, er);
   double raw[]; ArrayResize(raw, n * F);
   for(int t = 0; t < n; t++)
     {
      // vol: rolling std of eq * sqrt(252)
      double vol = CTA_RollStd(eq, n, 1, 0, t, p.vol_window);
      if(!CTA_IsNaN(vol)) vol *= MathSqrt(252.0);
      // corr: ((port_var * m / mean_var) - 1) / (m - 1), mean_var = mean over instruments of rolling var (skipna)
      double pv = CTA_RollVar(eq, n, 1, 0, t, p.corr_window);
      double mv = 0; int mc = 0;
      for(int j = 0; j < m; j++) { double v = CTA_RollVar(ret, n, m, j, t, p.corr_window); if(!CTA_IsNaN(v)) { mv += v; mc++; } }
      double rho = CTA_NAN();
      if(!CTA_IsNaN(pv) && mc > 0 && m > 1 && mv > 0) { rho = ((pv * m / (mv / mc)) - 1.0) / (m - 1); if(rho < -1) rho = -1; if(rho > 1) rho = 1; }
      // er: mean across instruments (skipna)
      double es = 0; int ec = 0;
      for(int j = 0; j < m; j++) { double v = er[t * m + j]; if(!CTA_IsNaN(v)) { es += v; ec++; } }
      double erm = (ec > 0) ? es / ec : CTA_NAN();
      // disp: rolling mean of cross-sectional std
      double disp = CTA_RollMean(xs, n, 1, 0, t, p.corr_window);
      // agree: share of instruments whose 40d and 252d signs agree (NaN counts as disagreement)
      double ag = 0;
      for(int j = 0; j < m; j++)
        {
         double fa = CTA_NAN(), sl = CTA_NAN();
         if(t >= 40 && !CTA_IsNaN(px[t * m + j]) && !CTA_IsNaN(px[(t - 40) * m + j]))  fa = CTA_Sign(px[t * m + j] / px[(t - 40) * m + j] - 1.0);
         if(t >= 252 && !CTA_IsNaN(px[t * m + j]) && !CTA_IsNaN(px[(t - 252) * m + j])) sl = CTA_Sign(px[t * m + j] / px[(t - 252) * m + j] - 1.0);
         if(!CTA_IsNaN(fa) && !CTA_IsNaN(sl) && fa == sl) ag += 1.0;
        }
      ag /= m;
      raw[t * F + 0] = vol; raw[t * F + 1] = rho; raw[t * F + 2] = erm; raw[t * F + 3] = disp; raw[t * F + 4] = ag;
     }
   ArrayResize(f, n * F);
   for(int t = 0; t < n; t++)
      for(int q = 0; q < F; q++)
         f[t * F + q] = (t == 0) ? CTA_NAN() : raw[(t - 1) * F + q];
  }

//============================================================ 4. portable k-means (cta/kmeans_portable.py)
ulong g_cta_rng;
void  CTA_RngSeed(const int seed)
  {
   g_cta_rng = ((ulong)seed * (ulong)0x9E3779B97F4A7C15) + (ulong)0x2545F4914F6CDD1D;
   if(g_cta_rng == 0) g_cta_rng = (ulong)0x2545F4914F6CDD1D;
  }
ulong CTA_RngNext()
  {
   ulong x = g_cta_rng;
   x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
   g_cta_rng = x;
   return x * (ulong)0x2545F4914F6CDD1D;
  }
double CTA_RngUniform() { return (double)(CTA_RngNext() >> 11) * (1.0 / 9007199254740992.0); }

double CTA_SqDist(const double &Z[], const int f, const int row, const double &c[], const int coff)
  {
   double s = 0;
   for(int q = 0; q < f; q++) { double d = Z[row * f + q] - c[coff + q]; s += d * d; }
   return s;
  }

// Returns inertia; centres in cent[k*f], labels in lab[n]. Consumes the global rng stream.
double CTA_KMeans(const double &Z[], const int n, const int f, const int k, const int n_init, const int max_iter,
                  double &cent[], int &lab[])
  {
   double best_inertia = -1; double best_c[]; int best_l[];
   double c[]; ArrayResize(c, k * f); int l[]; ArrayResize(l, n);
   double d2[]; ArrayResize(d2, n);
   for(int run = 0; run < n_init; run++)
     {
      // k-means++ (plain D2 sampling)
      int idx = (int)MathFloor(CTA_RngUniform() * n);
      for(int q = 0; q < f; q++) c[q] = Z[idx * f + q];
      for(int i = 0; i < n; i++) d2[i] = CTA_SqDist(Z, f, i, c, 0);
      for(int j = 1; j < k; j++)
        {
         double total = 0; for(int i = 0; i < n; i++) total += d2[i];
         double u = CTA_RngUniform();
         int pick = 0;
         if(total > 0)
           {
            double cs = 0; pick = n - 1;
            for(int i = 0; i < n; i++) { cs += d2[i]; if(cs > u * total) { pick = i; break; } }
           }
         for(int q = 0; q < f; q++) c[j * f + q] = Z[pick * f + q];
         for(int i = 0; i < n; i++) { double v = CTA_SqDist(Z, f, i, c, j * f); if(v < d2[i]) d2[i] = v; }
        }
      // Lloyd
      for(int i = 0; i < n; i++) l[i] = -1;
      for(int it = 0; it < max_iter; it++)
        {
         bool changed = false;
         for(int i = 0; i < n; i++)
           {
            int bj = 0; double bd = CTA_SqDist(Z, f, i, c, 0);
            for(int j = 1; j < k; j++) { double v = CTA_SqDist(Z, f, i, c, j * f); if(v < bd) { bd = v; bj = j; } }
            if(bj != l[i]) { changed = true; l[i] = bj; }
           }
         if(!changed) break;
         for(int j = 0; j < k; j++)
           {
            double s[]; ArrayResize(s, f); ArrayInitialize(s, 0.0); int cnt = 0;
            for(int i = 0; i < n; i++) if(l[i] == j) { for(int q = 0; q < f; q++) s[q] += Z[i * f + q]; cnt++; }
            if(cnt > 0) for(int q = 0; q < f; q++) c[j * f + q] = s[q] / cnt;
           }
        }
      // final assignment and inertia
      double inertia = 0;
      for(int i = 0; i < n; i++)
        {
         int bj = 0; double bd = CTA_SqDist(Z, f, i, c, 0);
         for(int j = 1; j < k; j++) { double v = CTA_SqDist(Z, f, i, c, j * f); if(v < bd) { bd = v; bj = j; } }
         l[i] = bj; inertia += bd;
        }
      if(best_inertia < 0 || inertia < best_inertia)
        {
         best_inertia = inertia;
         ArrayResize(best_c, k * f); ArrayCopy(best_c, c);
         ArrayResize(best_l, n);     ArrayCopy(best_l, l);
        }
     }
   ArrayResize(cent, k * f); ArrayCopy(cent, best_c);
   ArrayResize(lab, n);      ArrayCopy(lab, best_l);
   return best_inertia;
  }

int CTA_ElbowK(const double &inert[], const int &ks[], const int cnt)
  {
   if(cnt < 3) return ks[0];
   double ymin = inert[0], ymax = inert[0];
   for(int i = 1; i < cnt; i++) { if(inert[i] < ymin) ymin = inert[i]; if(inert[i] > ymax) ymax = inert[i]; }
   if(ymax - ymin == 0) return ks[0];
   int best = 0; double bestv = -1e300;
   for(int i = 0; i < cnt; i++)
     {
      double xn = (double)(ks[i] - ks[0]) / (double)(ks[cnt - 1] - ks[0]);
      double yn = (inert[i] - ymin) / (ymax - ymin);
      double y0 = (inert[0] - ymin) / (ymax - ymin), y1 = (inert[cnt - 1] - ymin) / (ymax - ymin);
      double chord = y0 + xn * (y1 - y0);
      double v = chord - yn;
      if(v > bestv) { bestv = v; best = i; }      // argmax takes the first maximum
     }
   return ks[best];
  }

//============================================================ 5. regime multiplier (learned ladder, auto-k, hysteresis)
// sleeve_ret: unlevered book return per day (NaN before the book is live).
// Output mult[n], 1.0 where the gate is inactive.
void CTA_RegimeMult(const double &px[], const int n, const int m, const datetime &dates[], const bool &is_rebal[],
                    const double &sleeve_ret[], const CTAParams &p, double &mult[])
  {
   const int F = 5;
   double f[]; CTA_RegimeFeatures(px, n, m, p, f);
   // valid rows (no NaN across the five features), in order
   int vrow[]; int nv = 0;
   for(int t = 0; t < n; t++)
     {
      bool ok = true;
      for(int q = 0; q < F; q++) if(CTA_IsNaN(f[t * F + q])) { ok = false; break; }
      if(ok) { ArrayResize(vrow, nv + 1); vrow[nv++] = t; }
     }
   ArrayResize(mult, n); for(int t = 0; t < n; t++) mult[t] = 1.0;
   if(nv <= p.regime_min_train) return;

   // forward sleeve return: sum of the next fwd_window days starting tomorrow (NaN if any NaN or truncated)
   double fwd[]; ArrayResize(fwd, n);
   for(int t = 0; t < n; t++)
     {
      double s = 0; bool ok = (t + p.regime_fwd_window < n);
      for(int i = 1; ok && i <= p.regime_fwd_window; i++) { double v = sleeve_ret[t + i]; if(CTA_IsNaN(v)) ok = false; else s += v; }
      fwd[t] = ok ? s : CTA_NAN();
     }

   double raw[]; ArrayResize(raw, n); for(int t = 0; t < n; t++) raw[t] = CTA_NAN();   // cluster rank per rebalance day
   // per-day learned map: mapk[t*K + rank] -> multiplier (K = k_max), valid where map_k[t] > 0
   int K = p.regime_k_max;
   double mapv[]; ArrayResize(mapv, n * K); int mapk[]; ArrayResize(mapk, n); ArrayInitialize(mapk, 0);

   double cent[]; int have_model = 0; int k_here = 0; int since_refit = 0;
   double mu[], sd[]; ArrayResize(mu, F); ArrayResize(sd, F);
   int rank_of[]; ArrayResize(rank_of, K);
   int last_map_t = -1;
   CTA_RngSeed(p.kmeans_seed);
   for(int t = 0; t < n; t++)
     {
      if(!is_rebal[t]) continue;
      // hist = valid rows up to and including t; need > min_train of them
      int nh = 0; while(nh < nv && vrow[nh] <= t) nh++;
      if(nh <= p.regime_min_train) continue;
      int ntrain = nh - 1;                                   // drop the current row
      if(have_model == 0 || since_refit >= p.regime_refit_every)
        {
         // z-score the training rows
         for(int q = 0; q < F; q++)
           {
            double s = 0, s2 = 0;
            for(int i = 0; i < ntrain; i++) { double v = f[vrow[i] * F + q]; s += v; s2 += v * v; }
            mu[q] = s / ntrain;
            double var = (s2 - ntrain * mu[q] * mu[q]) / (ntrain - 1);
            sd[q] = (var > 0) ? MathSqrt(var) : 1.0;
           }
         double Z[]; ArrayResize(Z, ntrain * F);
         for(int i = 0; i < ntrain; i++) for(int q = 0; q < F; q++) Z[i * F + q] = (f[vrow[i] * F + q] - mu[q]) / sd[q];
         // elbow over k_min..k_max: each fit consumes the rng stream in order, like the Python
         int ks[]; double inert[]; int cnt = 0;
         for(int k = p.regime_k_min; k <= p.regime_k_max; k++)
           {
            if(k >= ntrain) break;
            double c_[]; int l_[];
            // Python constructs a fresh KMeans(random_state=seed) per fit: reseed per fit
            CTA_RngSeed(p.kmeans_seed);
            double in = CTA_KMeans(Z, ntrain, F, k, p.kmeans_n_init, p.kmeans_max_iter, c_, l_);
            ArrayResize(ks, cnt + 1); ArrayResize(inert, cnt + 1); ks[cnt] = k; inert[cnt] = in; cnt++;
           }
         k_here = (cnt >= 3) ? CTA_ElbowK(inert, ks, cnt) : 3;
         int lab[];
         CTA_RngSeed(p.kmeans_seed);
         CTA_KMeans(Z, ntrain, F, k_here, p.kmeans_n_init, p.kmeans_max_iter, cent, lab);
         // rank clusters by the vol centroid (feature 0), ascending: rank 0 = calmest
         int order[]; ArrayResize(order, k_here);
         for(int j = 0; j < k_here; j++) order[j] = j;
         for(int a = 0; a < k_here; a++)                      // stable insertion sort (np.argsort default quicksort is not stable, but ties are measure-zero)
            for(int b = a + 1; b < k_here; b++)
               if(cent[order[b] * F] < cent[order[a] * F]) { int tmp = order[a]; order[a] = order[b]; order[b] = tmp; }
         for(int r = 0; r < k_here; r++) rank_of[order[r]] = r;
         have_model = 1; since_refit = 0;
        }
      since_refit++;
      // classify today (row vrow[nh-1] == t? the last valid row at or before t)
      int trow = vrow[nh - 1];
      double z[]; ArrayResize(z, F);
      for(int q = 0; q < F; q++) z[q] = (f[trow * F + q] - mu[q]) / sd[q];
      int bj = 0; double bd = CTA_SqDist(z, F, 0, cent, 0);
      for(int j = 1; j < k_here; j++) { double v = CTA_SqDist(z, F, 0, cent, j * F); if(v < bd) { bd = v; bj = j; } }
      raw[t] = rank_of[bj];
      // learned ladder: mean fwd return per rank over labelled days whose window closed before the cutoff
      double ladder_top = p.ladder_top, ladder_bottom = p.ladder_bottom;
      // Python: cutoff = dt - Timedelta(days=int(fwd_window * 1.45)); rows with date <= cutoff qualify
      datetime cutoff = dates[t] - (datetime)((int)MathFloor(p.regime_fwd_window * 1.45) * 86400);
      int cutoff_t = -1;
      for(int u = 0; u <= t; u++) { if(dates[u] <= cutoff) cutoff_t = u; else break; }
      double sum_[]; int cnt_[]; ArrayResize(sum_, k_here); ArrayResize(cnt_, k_here);
      ArrayInitialize(sum_, 0.0); ArrayInitialize(cnt_, 0);
      int nlab = 0;
      for(int u = 0; u <= cutoff_t && u <= t; u++)
        {
         if(CTA_IsNaN(raw[u]) || CTA_IsNaN(fwd[u])) continue;
         nlab++;                                   // Python checks len(hist) < 60 BEFORE filtering by k
         int r = (int)raw[u];
         if(r >= k_here) continue;
         sum_[r] += fwd[u]; cnt_[r]++;
        }
      mapk[t] = k_here;
      // default: rank i -> linspace(top, bottom, k)[i]
      for(int r = 0; r < k_here; r++) mapv[t * K + r] = CTA_Linspace(ladder_top, ladder_bottom, k_here, r);
      bool all_present = true;
      for(int r = 0; r < k_here; r++) if(cnt_[r] == 0) all_present = false;
      if(nlab >= 60 && all_present)
        {
         // order ranks by mean forward return, best first (sort_values descending; ties keep original order)
         double perf[]; ArrayResize(perf, k_here);
         for(int r = 0; r < k_here; r++) perf[r] = sum_[r] / cnt_[r];
         int ord[]; ArrayResize(ord, k_here); for(int r = 0; r < k_here; r++) ord[r] = r;
         for(int a = 0; a < k_here; a++)
            for(int b = a + 1; b < k_here; b++)
               if(perf[ord[b]] > perf[ord[a]]) { int tmp = ord[a]; ord[a] = ord[b]; ord[b] = tmp; }
         for(int jx = 0; jx < k_here; jx++) mapv[t * K + ord[jx]] = CTA_Linspace(ladder_top, ladder_bottom, k_here, jx);
        }
      last_map_t = t;
     }
   // forward-fill raw labels, then hysteresis
   double cur = CTA_NAN(), prev = CTA_NAN(); int run = 0;
   double stable[]; ArrayResize(stable, n);
   double last_raw = CTA_NAN();
   for(int t = 0; t < n; t++)
     {
      if(!CTA_IsNaN(raw[t])) last_raw = raw[t];
      double v = last_raw;
      if(CTA_IsNaN(v) || CTA_IsNaN(prev) || v != prev) { run = 1; prev = v; } else run++;
      if(!CTA_IsNaN(v) && (CTA_IsNaN(cur) || run >= p.regime_hysteresis)) cur = v;
      stable[t] = cur;
     }
   // apply the map in force on each date (forward-filled)
   int mt = -1;
   for(int t = 0; t < n; t++)
     {
      if(mapk[t] > 0) mt = t;
      if(mt < 0 || CTA_IsNaN(stable[t])) { mult[t] = 1.0; continue; }
      int r = (int)stable[t];
      mult[t] = (r < mapk[mt]) ? mapv[mt * K + r] : 1.0;
     }
  }

double CTA_Linspace(const double top, const double bottom, const int k, const int i)
  {
   if(k <= 1) return top;
   return top + (bottom - top) * (double)i / (double)(k - 1);
  }

// Monthly calendar rebalancing: the first bar of each new month (pandas to_period("M") change).
void CTA_MonthlyRebal(const datetime &dates[], const int n, bool &is_rebal[])
  {
   ArrayResize(is_rebal, n);
   MqlDateTime a, b;
   for(int t = 0; t < n; t++)
     {
      if(t == 0) { is_rebal[t] = false; continue; }
      TimeToStruct(dates[t], a); TimeToStruct(dates[t - 1], b);
      is_rebal[t] = (a.year != b.year || a.mon != b.mon);
     }
  }

//============================================================ 6. the book
// Produces target weights w[t*m+j] (fraction of NAV, leverage-scaled, the
// HELD weight from each rebalance) and leverage lev[t].
void CTA_BuildTargets(const double &px[], const int n, const int m, const datetime &dates[], const int &bucket[],
                      const int n_buckets, const bool &is_rebal[], const CTAParams &p,
                      double &w[], double &lev[], double &mult_out[])
  {
   // Python: returns = prices.pct_change().fillna(0.0) for sizing, HRP and the book
   double ret[]; CTA_PctChange(px, n, m, 1, ret);
   for(int i = 0; i < n * m; i++) if(CTA_IsNaN(ret[i])) ret[i] = 0.0;
   double score[]; CTA_TsmomScore(px, n, m, p, score);
   double hrp[];   CTA_HrpScale(ret, n, m, bucket, n_buckets, is_rebal, p, hrp);
   // instrument vol (lagged one day) and Tier-1 weight
   double iw[]; ArrayResize(iw, n * m);
   for(int t = 0; t < n; t++)
      for(int j = 0; j < m; j++)
        {
         double v = (t >= 1) ? CTA_RollStd(ret, n, m, j, t - 1, p.vol_window) : CTA_NAN();
         double s = score[t * m + j];
         double x = CTA_NAN();
         if(!CTA_IsNaN(v) && !CTA_IsNaN(s) && v > 0) x = s * p.inst_target_vol * hrp[t * m + j] / (v * MathSqrt(252.0));
         if(CTA_IsNaN(x)) x = 0.0;
         if(x > p.max_inst_weight) x = p.max_inst_weight;
         if(x < -p.max_inst_weight) x = -p.max_inst_weight;
         iw[t * m + j] = x;
        }
   // unlevered book return with the warm-up guard (NaN until the book has ever been live)
   double pru[]; ArrayResize(pru, n);
   bool live = false;
   for(int t = 0; t < n; t++)
     {
      double s = 0; bool any = false;
      for(int j = 0; j < m; j++)
        {
         if(!CTA_IsNaN(score[t * m + j])) any = true;
         s += ret[t * m + j] * iw[t * m + j];
        }
      if(any) live = true;
      pru[t] = live ? s : CTA_NAN();
     }
   // regime multiplier on the Tier-2 target
   double mult[]; CTA_RegimeMult(px, n, m, dates, is_rebal, pru, p, mult);
   ArrayResize(mult_out, n); ArrayCopy(mult_out, mult);
   // Tier-2 leverage at rebalance: target / lagged portfolio vol, capped
   ArrayResize(lev, n); ArrayResize(w, n * m);
   double cur_lev = 0.0; double cur_w[]; ArrayResize(cur_w, m); ArrayInitialize(cur_w, 0.0);
   for(int t = 0; t < n; t++)
     {
      if(is_rebal[t])
        {
         // Python: (target / port_vol_lagged).clip(upper=max_leverage).fillna(0.0), and the
         // rebalance ALWAYS resets the book on a rebalance day (NaN vol -> leverage 0).
         double pv = (t >= 1) ? CTA_RollStd(pru, n, 1, 0, t - 1, p.vol_window) : CTA_NAN();
         double L = 0.0;
         if(!CTA_IsNaN(pv)) L = (pv > 0) ? p.base_target_vol * mult[t] / (pv * MathSqrt(252.0)) : p.max_leverage;
         if(L > p.max_leverage) L = p.max_leverage;
         cur_lev = L;
         for(int j = 0; j < m; j++) cur_w[j] = iw[t * m + j];
        }
      lev[t] = cur_lev;
      for(int j = 0; j < m; j++) w[t * m + j] = cur_lev * cur_w[j];
     }
  }
