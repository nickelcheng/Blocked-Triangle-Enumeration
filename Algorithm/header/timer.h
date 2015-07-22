#include<sys/time.h>
#include<cstdio>

#define cntTime(st,ed)\
((double)ed.tv_sec*1000000+ed.tv_usec-(st.tv_sec*1000000+st.tv_usec))/1000

#define timerInit(n)\
struct timeval st[n], ed[n];

#define timerStart(n)\
gettimeofday(&st[n], NULL);

#define timerEnd(tar, n)\
gettimeofday(&ed[n], NULL);\
fprintf(stderr, "%s: %.3lf ms\n", tar, cntTime(st[n],ed[n]));
//fprintf(stderr, " %.3lf", cntTime(st[n],ed[n]));

