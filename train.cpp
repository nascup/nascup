#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h> 
#include <omp.h>
#include <iostream>
#include <ctype.h>
#include <sys/time.h>
#include "ctm.h"
using namespace std;

#ifdef PROTEIN
    int D = 3;
    int D1 = 9261;
#else
    int D = 6;
    int D1 = 15625;
#endif

int method = KT;
double sdcnt = 0;
bool markov = true;
bool full = false;

int m;
CTM *mdl;
char acc[128], acc_prev[128];
char seq[1 << 14];

Tree *mem;
int (*c)[A];
unsigned long int (*t)[A];
int (*tt)[A];
double *pe, *pm;

FILE *ref_f, *mdl_f;
char ref_fn[128] = "train.fasta";
char mdl_fn[128] = "model.ctm";

void ctm(int context)
{
    if(mem[context].div != -2)
        return;

    mem[context].div = -1;
    mem[context].p = pe[context];
    int i, j;
    for(j=0, i=0; i < A; ++i)
        j += (tt[context][i] == 0);
    if(j >= A - 1) return;

    int subcontext[A];
    double prod, p_m[A], p_after[A + 1];
    for(i = 0; i < D; ++i)
    {
        if(context / pw[i] % (A + 1) != 0) continue;
        for(j = 0; j < A; ++j)
            p_m[j] = pm[subcontext[j] = context + (j + 1) * pw[i]];
        p_after[A] = 0;
        for(j = A - 1; j >= 0; --j)
            p_after[j] = p_m[j] + p_after[j + 1];
        prod = 0;
        for(j = 0; j < A; ++j)
        {
            if(prod + p_after[j] <= mem[context].p) break;
            ctm(subcontext[j]);
            prod += mem[subcontext[j]].p;
        }
        if(j == A && prod > mem[context].p)
            mem[context].div = i, mem[context].p=prod;
        if(markov)
            break;
    }
}

void gather(CTM *mdl, int context)
{
    int div, i;
    div = mem[context].div;
    if(div == -2) return;
    if(div == -1)
    {
        Leaf leaf;
        leaf.context = context;
        for(i = 0; i < A; ++i)
            leaf.param[i] = tt[context][i];
        mdl->leaves.push_back(leaf);
    }
    else
        for(i = 0; i < A; ++i)
            gather(mdl, context + (i + 1) * pw[div]);
}

void find_model(CTM *mdl)
{
    int i;
    for(i = 0; i < pw[D]; ++i)
        mem[i].div = -2;
    ctm(0);
    gather(mdl, 0);
}

int main(int argc, char *argv[])
{
    int eof,i,j;
    int opt;
    int errflag = 0;
    char est[128];
    while((opt = getopt(argc, argv, "i:o:m:d:s:ch")) != -1)
        switch(opt)
        {
        case 'i':
            strcpy(ref_fn, optarg);
            break;
        case 'o':
            strcpy(mdl_fn, optarg);
            break;
        case 'm':
            strcpy(est, optarg);
            for(int c = 0; est[c]; ++c)
                est[c] = tolower(est[c]);
            if(strcmp(est, "ml") == 0)
                method = ML;
            else if(strcmp(est, "mlc") == 0)
                method = ML, sdcnt = 0.5;
            else if(strcmp(est, "kt") == 0)
                method = KT;
            else if(strcmp(est, "zr") == 0)
                method = ZR;
            else if(strcmp(est, "kmer") == 0)
                full = true;
            else
                errflag = 1;
            break;
        case 'd':
            D = atoi(optarg);
            if(D < 0 || D > 12)
                errflag = 1;
            break;
        case 's':
            sdcnt = atof(optarg);
            break;
        case 'c':
            markov = false;
            break;
        case '?':
        case 'h':
        default:
            errflag = 1;
        }

    if((ref_f = fopen(ref_fn, "r")) == NULL)
        errflag = 1;
    if((mdl_f = fopen(mdl_fn, "w")) == NULL)
        errflag = 1;

    if(errflag)
        exit(1);


    fprintf(mdl_f, "MAXD\t%d\n\n", D);
    ctm_init();

    mem = (Tree *)malloc(sizeof(Tree) * D1);
    c = (int(*)[A])malloc(sizeof(int) * D1 * A);
    t = (unsigned long int (*)[A])malloc(sizeof(unsigned long int)* D1 * A);
    pe=(double *)malloc(sizeof(double) * D1);
    pm=(double *)malloc(sizeof(double) * D1);
    tt=(int (*)[A])malloc(sizeof(int) * D1 * A);
    
    int nseq = 0;
    for(m = 0; ;)
    {
        eof = (fscanf(ref_f, " >%s%s", acc, seq) != 2);
        if(strcmp(acc_prev, acc) != 0 || eof)
        {
            if(m)
            {
                mdl = new CTM(acc_prev);
                for(i = 0; i < D1; ++i)
                    for(j = 0; j < A; ++j)
                        tt[i][j] = (int)t[i][j];

                if(!full)
                {
                    if(method==ML)
                        #pragma omp parallel for
                        for(i = 0; i < D1; ++i)
                            pe[i] = ml(tt[i]), pm[i] = ml_max(tt[i]);
                    else if(method==KT)
                        #pragma omp parallel for
                        for(i = 0; i < D1; ++i)
                            pe[i] = kt(tt[i]), pm[i]=kt_max(tt[i]);
                    else if(method==ZR)
                        #pragma omp parallel for
                        for(i = 0; i < D1; ++i)
                            pe[i] = zr(tt[i]), pm[i] = zr_max(tt[i]);
                    find_model(mdl);
                }
                else
                {
                    for(i = 0; i < D1; ++i)
                    {
                        for(j = 0; j < D && (i / pw[j] % (A + 1) !=0);++j);
                        if(j == D)
                        {
                            Leaf leaf;
                            leaf.context=i;
                            for(j = 0; j < A; ++j)
                                leaf.param[j] = tt[i][j];
                            mdl->leaves.push_back(leaf);
                        }
                    }
                }
                mdl->save(mdl_f);
            }
            ++m;
            memset(t, 0, sizeof(unsigned long int) * D1 * A);
            memset(tt, 0, sizeof(int) * D1 * A);
            memset(pe, 0, sizeof(double) * D1);
            memset(pm, 0, sizeof(double) * D1);
            strcpy(acc_prev, acc);
            nseq = 0;
        }
        nseq++;
        if(eof){--m; break;}
        count(seq, c);

        #pragma omp parallel for private(j)
        for(i = 0; i < D1; ++i)
            for(j = 0; j < A; ++j)
                t[i][j] += c[i][j];
    }

    fclose(ref_f);
    fclose(mdl_f);
    return 0;
}
