#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <vector>
#include <omp.h>
using namespace std;

//#define PROTEIN

#ifdef PROTEIN
    const int A = 20;
    const char chr[A + 2] = "NGAVLIPFYWSTCMNQKRHDE";
    const int pw[] = {1, 21, 441, 9261, 194481, 4084101};
    extern int D;
    extern int D1;
#else
    const int A = 4;
    const char chr[A + 2] = "NACGT";
    const int pw[] = {1, 5, 25, 125, 625, 3125, 15625, 78125, 390625,
                      1953125, 9765625, 48828125, 244140625};
    extern int D;
    extern int D1;
#endif

enum METHOD {ML, MML, KT, ZR, EUC};

const int MAX_CNT = 1<<25;
const double inf = 1e100;

extern double sdcnt;

int chr2i[256];
int *ld, *ns, *sp, *inner;
int ones[1 << A];
double fct0[MAX_CNT], fct1[MAX_CNT];
double lg2, lg0[MAX_CNT], lg1[MAX_CNT];

void count(char *s, int (*t)[A])
{
    int I, i, j, k, h, context;
    memset(t, 0, sizeof(int) * D1 * A);
    context=0;
    for(i = 0; i < D; ++i)
    {
        j = chr2i[(unsigned char)s[i]];
        context = context * (A + 1) + j;
    }
    for(; s[i]; ++i)
    {
        j = chr2i[(unsigned char)s[i]];
        k = chr2i[(unsigned char)s[i - D]];
        t[context][j - 1]++;
        context = context * (A + 1) - k * pw[D] + j;
    }

    for(h = 0; h < D; ++h)
        #pragma omp parallel for private(i, j, k)
        for(I = sp[h]; I < sp[h + 1]; ++I)
        {
            i = inner[I];
            for(j = 1; j <= A; ++j)
                for(k =0; k<A; ++k)
                    t[i][k] += t[i + j * pw[ld[i]]][k];
        }
}

struct Tree{int div; double p;};
struct Leaf{int context; int param[A];};

class CTM
{
    public:
        char acc[128];
        vector<Leaf> leaves;

        CTM(){}
        CTM(char *acc){strcpy(this->acc,acc); leaves.clear();}
        int load(FILE *fp);
        void save(FILE *fp);
};

int CTM::load(FILE *fp)
{
    int ln, i, j;
    Leaf leaf;
    if(fscanf(fp," ACC\t%s NLEAVES\t%d", acc, &ln) != 2)
        return 0;
    leaves.clear();
    for(i = 0; i < ln; ++i)
    {
        if(fscanf(fp, "%d", &leaf.context) != 1)
            return 0;
        for(j = 0; j<A; ++j)
            if(fscanf(fp, " %d", &leaf.param[j]) != 1)
                return 0;
        leaves.push_back(leaf);
    }
    return 1;
}

void CTM::save(FILE *fp)
{
    int ln, i, j;
    fprintf(fp, "ACC\t%s\n", acc);
    fprintf(fp, "NLEAVES\t%d\n", ln = leaves.size());
    for(i = 0; i < ln; ++i)
    {
        fprintf(fp, "%d", leaves[i].context);
        for(j = 0; j < A; ++j)
            fprintf(fp," %d", leaves[i].param[j]);
        fprintf(fp, "\n");
    }
    fprintf(fp,"\n");
}

double ml(int x[A])
{
    int i, s = 0;
    double mlp = 0;
    for(i = 0; i < A; ++i)
        if(x[i])
            s += x[i], mlp += x[i] * lg0[x[i]];
    if(s)
        mlp -= s * lg1[s];
    return mlp;
}

double ml_max(int x[A])
{
    int i;
    double mlp = 0;
    for(i = 0; i < A; ++i)
        if(x[i])
            mlp += x[i] * (lg0[x[i]] - lg1[x[i]]);
    return mlp;
}

double ml_infer(int x[A],int y[A])
{
    int i, s = 0, s1 = 0;
    double mlp = 0;
    for(i = 0; i < A; ++i)
    {
        if(sdcnt == 0 && y[i] == 0 && x[i] > 0)
            return -inf;
        s1 += y[i];
        if(x[i])
            s += x[i], mlp += x[i] * lg0[y[i]];
    }
    if(sdcnt != 0 || s1)
        mlp -= s * lg1[s1];
    return mlp;
}


double kt(int x[A])
{
    int i, s = 0;
    double ktp = 0;
    for(i = 0; i < A; ++i)
        s += x[i], ktp += fct1[x[i]];
    ktp -= (A & 1) ? fct1[s + (A - 1) / 2] : fct0[s - 1 + A / 2];
    return ktp;
}

double kt_max(int x[A])
{
    int i;
    double ktp = 0;
    for(i = 0; i < A; ++i)
        ktp += fct1[x[i]] - ((A & 1) ? fct1[x[i] + (A - 1) / 2]
                                     : fct0[x[i] - 1 + A / 2]);
    return ktp;
}

double kt_infer(int x[A], int y[A])
{
    int i, s = 0, s1 = 0;
    double ktp = 0;
    for(i = 0; i < A; ++i)
        s +=x[i], s1 += y[i], ktp += fct1[y[i] + x[i]] - fct1[y[i]];
    ktp -= (A & 1) ? fct1[s1 + s + (A - 1) / 2] - fct1[s1 + (A - 1) / 2]
                   : fct0[s1 + s - 1 + A / 2] - fct0[s1 - 1 + A / 2];
    return ktp;
}

inline double logsumexp(double x, double y)
{
    return x > y ? x + log(1 + exp(y - x)) : y + log(1 + exp(x - y));
}

inline double prod(int t, int a)
{
    return (a & 1) ? fct1[t + (a - 1) / 2] - fct1[(a - 1) / 2]
                   : fct0[t - 1 + a / 2] - fct0[a / 2 - 1];
}

double zr(int x[A])
{
    int i, j, k, s, z;
    double np, dp;
    z = 0; s = 0; np = 0;
    for(i = 0; i < A; ++i)
        if(x[i] == 0)
            ++z;
        else
            s += x[i], np += fct1[x[i]];
    dp = lg2 - prod(s, A);
    for(j = 0; j < (1 << z) - 1; ++j)
        k = (A - z) + ones[j], dp = logsumexp(dp, -prod(s, k));
    dp -= A * lg2;
    return np + dp;
}

double zr_max(int x[A])
{
    int i, y[A];
    double zrp = 0;
    for(i = 0; i < A; ++i)
        y[i] = 0;
    for(i = 0; i < A; ++i)
        y[0] = x[i], zrp += zr(y);
    return zrp;
}

double zr_infer(int x[A], int y[A])
{
    int Y[A];
    int i, j, k, s, s1, ss1, z;
    double np, dp;
    z = 0; s = s1 = ss1 = 0; np = 0;
    for(i = 0; i < A; ++i)
        if(x[i] == 0)
            Y[z++] = y[i], ss1 += y[i];
        else
            s += x[i], s1 += y[i], np += fct1[y[i] + x[i]] - fct1[y[i]];
    if(z == A)
        return 0;
    dp = lg2 - (prod(ss1 + s1 + s, A) - prod(ss1 + s1, A));
    for(j = 0; j< (1 << z) - 1; ++j)
    {
        ss1 = 0;
        for(i = 0; i < A; ++i)
            ss1 += (j & (1 << i)) ? Y[i] : 0;
        k = (A - z) + ones[j];
        dp = logsumexp(dp, -(prod(ss1 + s1 + s, k) - prod(ss1 + s1, k)));
    }
    dp -= A * lg2;
    return np + dp;
}

void ctm_init()
{
    int i, j;
    for(i = 0; i < A + 1; ++i)
        chr2i[(unsigned char)chr[i]] = i;

    inner = (int *)malloc(sizeof(int) * D1);
    sp = (int *)malloc(sizeof(int) * (D + 1));
    memset(sp, 0, sizeof(int) * (D + 1));
    ld = (int *)malloc(sizeof(int) * D1);
    ns = (int *)malloc(sizeof(int) * D1);
    ns[0] = 0;
    for(i = 0; i < pw[D]; ++i)
    {
        j = i / (A + 1);
        if(i % (A + 1) == 0)
            ld[i] = 0, ns[i] = ns[j];
        else
            ld[i] = ld[j] + 1, ns[i] = ns[j] + 1;
        if(ns[i] < D)
            sp[D - ns[i] - 1]++;
    }
    for(i = 1; i <= D; ++i)
        sp[i] += sp[i - 1];
    for(i = 0; i < pw[D]; ++i)
        if(ns[i] < D)
            inner[--sp[D - ns[i] - 1]] = i;

    lg2 = log(2);
    fct0[0] = fct1[0] = 0;
    for(i = 1; i < MAX_CNT; ++i)
        fct0[i] = fct0[i - 1] + log(i),
        fct1[i] = fct1[i - 1] + log(i - 0.5);

    for(i = 0; i < MAX_CNT; ++i)
        lg0[i] = log(i + sdcnt),
        lg1[i] = log(i + sdcnt*A);

    ones[0] = 0;
    for(i = 1; i < (1 << A); ++i)
        ones[i] = ones[i>>1] + (i&1);
}
