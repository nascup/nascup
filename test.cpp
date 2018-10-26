#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h> 
#include <vector>
#include <omp.h>
#include <iostream>
#include <sys/time.h>
#include "ctm.h"
using namespace std;

int D;
int D1;
int method = KT;
double sdcnt = 0;
double cut_off = 0;

int m, n;
CTM *mdl;
vector<CTM *> mdl_list;
char name[128];
char seq[1 << 14];

int (*c)[A];
vector<double> pt;
double pt_max;
int match;

FILE *mdl_f, *inp_f, *out_f;
char mdl_fn[128] = "model.ctm";
char inp_fn[128] = "test.fasta";
char out_fn[128] = "result.out";

bool bnb = true;

inline double square(double x){return x * x;}

double prob(CTM *mdl, int (*t)[A])
{
    int ln, i;
    double p = 0, pp;
    ln = mdl->leaves.size();
    if(method == ML)
        for(i = 0; i < ln; ++i)
        {
            pp = ml_infer(t[mdl->leaves[i].context], mdl->leaves[i].param);
            if(pp == -inf)
                return -inf;
            p += pp;
            if(p < pt_max && bnb) return -inf;
        }
    else if(method == KT)
        for(i = 0; i < ln; ++i)
        {
            p += kt_infer(t[mdl->leaves[i].context], mdl->leaves[i].param);
            if(p < pt_max && bnb) return -inf;
        }
    else if(method == ZR)
        for(i = 0; i < ln; ++i){
            p += zr_infer(t[mdl->leaves[i].context], mdl->leaves[i].param);
            if(p < pt_max && bnb) return -inf;
        }
    return p;
}

int main(int argc, char *argv[])
{
    int i;
    int opt;
    int errflag = 0;     
    char est[128];
    while((opt = getopt(argc, argv, "c:i:o:m:s:bh")) != -1)
        switch(opt)
        {
        case 'c':
            strcpy(mdl_fn, optarg);
        case 'i':
            strcpy(inp_fn, optarg);
            break;
        case 'o':
            strcpy(out_fn, optarg);
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
            else
                errflag = 1;
            break;
        case 's':
            sdcnt = atof(optarg);
            break;
        case '?':
        case 'h':
        default:
            errflag = 1;
        }

    if((mdl_f = fopen(mdl_fn,"r")) == NULL)
        errflag = 1;
    if((inp_f = fopen(inp_fn,"r")) == NULL)
        errflag = 1;
    if((out_f = fopen(out_fn, "w")) == NULL)
        errflag = 1;

    if(errflag)
        exit(1);

    if(fscanf(mdl_f, " MAXD\t%d",&D) != 1)
        exit(1);
    D1 = pw[D];

    ctm_init();
    for(m = 0; ; )
    {
        mdl = new CTM();
        if(mdl->load(mdl_f))
            mdl_list.push_back(mdl), pt.push_back(0), m++;
        else
            break;
    }

    double *prb = new double[m];
    int seqlen = 0;
    c = (int (*)[A])malloc(sizeof(int) * D1 * A);
    for(n = 0; fscanf(inp_f, " >%s%s",name, seq) == 2; )
    {
        n++;
        count(seq, c);
        pt_max = -inf;
        match = -1;
        seqlen = strlen(seq);
        if(seqlen > 0)
        {
            #pragma omp parallel for
            for(i=0;i<m;++i){
                pt[i]=prob(mdl_list[i],c);
                prb[i] = pt[i];
                if(pt[i]>pt_max)
                    pt_max=pt[i],match=i;
            }
        }
        fprintf(out_f, "%s\t%s\n", name, match == -1 ? "0" : mdl_list[match]->acc);
    }

    fclose(mdl_f);
    fclose(inp_f);
    fclose(out_f);
    return 0;
}
