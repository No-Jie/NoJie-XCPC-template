```c++
i>>j&1 //判断i的第j位是否为1、取出i的第j位
i&1 //判断i的最后一位是否为1
gcd(x,y,z)=gcd(x,y-x,z-y)
```

> 异或是不进位的加法

素数：1e9+7，998244353，99991



[洛谷 P4779](https://vjudge.net/problem/洛谷-P4779/origin)[单源最短路径（标准版）](https://vjudge.net/contest/507187#problem/C)

[洛谷 P1339](https://vjudge.net/problem/洛谷-P1339/origin)[Heat Wave G](https://vjudge.net/contest/507187#problem/E)

[洛谷 P1346](https://vjudge.net/problem/洛谷-P1346/origin)[电车](https://vjudge.net/contest/507187#problem/F)边权为0或1，求单源最短路

[HDU 1285](https://vjudge.net/problem/HDU-1285/origin)[确定比赛名次](https://vjudge.net/contest/507187#problem/H)拓扑排序



### 关闭流同步

```c++
ios::sync_with_stdio(false), cin.tie(0);
```

### 快读

```c++
int read() {
	int sum=0,p=1;
	char c=getchar();
	while(c<'0'||c>'9') {
		if(c=='-') p=-1;
		c=getchar();
	}
	while(c>='0'&&c<='9') {
		sum=(sum*10+c-'0')%mod;5
		c=getchar();
	}
	return sum*p;
}
```

### 文件读写

```c++
freopen("a.in","r",stdin);
freopen("a.out","w",stdout);
```

# 二分

### 整数 

```c++
int l = 1, r = n, mid;
while (l <= r){
    mid = (l + r) / 2;
    if (check(mid)) l = mid + 1;
    else r = mid - 1;
}
int ans = l - 1;
```

### 小数

```c++
double l = a, r = b, mid, eps = 1e-6;
while (r - l >= eps){
    mid = l + (r - l) / 2.;
    if (f(mid) >= -eps)r = mid;
    else l = mid;
}
cout << l;
```

# 三分

```c++
double l, r, lsec, rsec, eps = 1e-8, mid;
while (r - l >= eps){
    mid = (l + r) / 2;
    lsec = mid - eps;
    rsec = mid + eps;
    if (f(lsec) < f(rsec)) l = mid;
    else r = mid;
}
printf("%.5lf\n", l);
```

# 整除分块

```c++
for(int l=a,r=a;l<=b;l=r+1)
{
    r=b/(b/l);
    int k=b/l;
}

 for(int l=a,r=a;l<=b;l=r)
 {
     r=b/(b/l);
     r+=(r==l);
     int k=b/l;
 }
```

# 离散化

![image-20220826225312756](C:\Users\巷子里的童年\AppData\Roaming\Typora\typora-user-images\image-20220826225312756.png)

![image-20220826225318201](C:\Users\巷子里的童年\AppData\Roaming\Typora\typora-user-images\image-20220826225318201.png)

# STL

### **重载运算符**

```c++
bool operator<(const Point& a) const{ return x < a.x; }//重载小于号 
friend bool operator>(Point a, Point b){ return a.x > b.x; }//重载大于号 
```

### **priority_queue**

大顶堆定义小于号，小顶堆定义大于号，greater

```c++
struct Point{
	int x, y;
	friend bool operator<(Point a, Point b){return a.x < b.x;}
};
priority_queue<Point> a; //大顶堆

struct Point {
	int x, y;
	friend bool operator>(Point a, Point b){return a.x > b.x;}
}; 
priority_queue<Point, vector<Point>, greater<Point> > a; //小顶堆
```

# 数据结构

### ST表

RMQ：区间最大/最小值

ST表可解决RMQ，区间gcd等问题， $ O(nlogn) $ 预处理， $ O(1) $ 询问

令 $f(i,j)$ 表示区间 $[i,i+2^j-1]$  的最大值。显然 $f(i,0)=a_i$ 。

状态转移方程： $ f(i,j)=\max(f(i,j-1),f(i+2^{j-1},j-1)) $ 。

对于每个询问 $  [l,r] $ ，我们把它分成两部分： $ f[l,s] $  与  $ f[r-2^s+1,s] $ ，其中  $ s=\left\lfloor\log_2(r-l+1)\right\rfloor $ 。两部分的结果的最大值就是回答。

log 函数预处理：

 $  $ 
\left\{\begin{aligned}
Logn[1] &=0, \\
Logn\left[i\right] &=Logn[\frac{i}{2}] + 1.
\end{aligned}\right.
 $  $ 

### 并查集

```c++
int find(int x){
	if (fa[x]==x) return x;
	else return fa[x]=find(fa[x]);
} 
void Union(int x,int y){
	int fx=find(x),fy=find(y);
	if (fx==fy)return;
	fa[fx]=fy;
}
for (int i=1;i<=n;i++) fa[i]=i;
```

### 线段树

区间修改，区间查询

```c++
#include<bits/stdc++.h>
using namespace std;
#define mid ((l+r)>>1)
#define lson rt<<1,l,mid
#define rson rt<<1|1,mid+1,r
#define N 10010

typedef long long ll;
const int MOD = 1e9 + 7;
int sum[N * 4], lazy[N * 4], a[N];

void pushup(int rt) { sum[rt] = sum[rt << 1] + sum[rt << 1 | 1]; }

void pushdown(int rt, int l, int r) {
    if (!lazy[rt]) return;
    lazy[rt << 1] += lazy[rt];
    sum[rt << 1] += lazy[rt] * (mid - l + 1);

    lazy[rt << 1 | 1] += lazy[rt];
    sum[rt << 1 | 1] += lazy[rt] * (r - mid);

    lazy[rt] = 0;
}

void build(int rt, int l, int  r) {
    if (l == r) {
        sum[rt] = a[l];
        return;
    }
    build(lson);
    build(rson);
    pushup(rt);
}

int query(int rt, int l, int r, int L, int R) {//询问L到R区间和
    if (l == L && r == R)
        return sum[rt] ;
    pushdown(rt, l, r);
    if (mid < L)
        return query(rson, L, R) ;
    else if (mid >= R)
        return query(lson, L, R) ;
    else
        return query(lson, L, mid) + query(rson, mid + 1, R);
}

void update(int rt, int l, int r, int L, int R, int val) {//L到R每个数都加val
    if (l == L && r == R) {
        sum[rt] += val * (r - l + 1);
        lazy[rt] += val;
        return;
    }
    pushdown(rt, l, r);
    if (mid < L)
        update(rson, L, R, val);
    else if (mid >= R)
        update(lson, L, R, val);
    else {
        update(lson, L, mid, val);
        update(rson, mid + 1, R, val);
    }
    pushup(rt);
}

int main() {
    ios::sync_with_stdio(false), cin.tie(0);
    int m, n;
    cin >> n >> m;
    for (int i = 1; i <= n; i++)
        cin >> a[i];
    build(1, 1, n);

    while (m--) {
        int op, x, y;
        cin >> op >> x >> y;
        if (op == 1) {
            int k;
            cin >> k;
            update(1, 1, n, x, y, k);
        }
        else if (op == 2)
            cout << query(1, 1, n, x, y);
    }
    return 0;
}
```

### 树状数组

单点修改，区间查询

普通树状数组维护的信息及运算要满足 **结合律** 且 **可差分**

```c++
int n,t[200010];
int lowbit(int x){return x&(-x);}
int find(int x){//区间查询
	int sum = 0;
	for (; x ; x -= lowbit(x)) sum += t[x];
    return sum;
}
void add(int x, int k){//将[x,n]区间内每个数都加k，单点修改
	for (; x <= n; x += lowbit(x)) t[x] += k;
}
//建树：n次单点修改,o(nlogn)
```

![image-20220826231648679](C:\Users\巷子里的童年\AppData\Roaming\Typora\typora-user-images\image-20220826231648679.png)

# 动态规划

### 高位前缀和/sos dp

求满足a^b=a+b的数对个数

```c++
for (int i = 1; i <= n; i++) {
    cin >> a[i];
    sum[a[i]]++;
}
int mx = (1 << 17) - 1;
for (int j = 0; j < 17; j++) {//枚举当前处理哪一维度
    for (int i = 0; i <= mx; i++) {
        if (i & (1 << j)) {
            sum[i] += sum[i ^ (1 << j)];//如果该维度为1，统计上该维度为0的前缀和
        }
    }
}
for (int i = 1; i <= n; i++) {
    ans += sum[a[i] ^ mx];
}
cout << ans << "\n";
```

### 01背包

有 N件物品和一个容量为 M 的背包。第 i件物品的重量是Wi，价值是 Di。求解将哪些物品装入背包可使这些物品的重量总和不超过背包容量，且价值总和最大。

```c++
for (int i=1;i<=n;i++){
    for (int j=m;j>=w[i];j--){
        dp[j]=max(dp[j],dp[j-w[i]]+d[i]);
    }
} 
```

### 完全背包

```c++
for (int i=1;i<=n;i++){
    for (int j=w[i];j<=m;j++){
        dp[j]=max(dp[j],dp[j-w[i]]+d[i]);
    }
} 
```

### 树形dp

##### 没有上司的舞会

f[x][0]表示以x为根的子树，且x不参加舞会的最大快乐值

f[x][1]表示以x为根的子树，且x参加了舞会的最大快乐值

```c++
#include<bits/stdc++.h>
#include <iostream>
#include <algorithm>
#include<vector>
using namespace std;
int n, r[6003], l, k, dp[6003][2];
vector<int>point[6003];
int dfs(int rt, int state) {//以rt为根的子树，选择/不选rt获得的最大快乐值
    if (dp[rt][state] != -1) return dp[rt][state];
    int ans = 0;
    if (state == 1) {
        for (auto it : point[rt])
            ans += dfs(it, 0);
        return dp[rt][state] = ans + r[rt];
    }
    else {
        for (auto it : point[rt])
            ans += max(dfs(it, 0), dfs(it, 1));
        return dp[rt][state] = ans;
    }
}
int main() {
    cin >> n;
    for (int i = 1; i <= n; i++) {
        cin >> r[i];
    }
    for (int i = 1; i < n; i++) {
        cin >> l >> k;
        point[k].push_back(l);
    }
    memset(dp, -1, sizeof(dp));
    int res = 0;
    for (int i = 1; i <= n; i++) {
        res =max(res, max(dfs(i, 1), dfs(i, 0)));
    }
    cout << res << "\n";
}
```

### 树上背包

树上背包=树形dp+背包问题

##### 选课

设f(u,i,j)表示以u号点为根的子树中，已经遍历了u号点的前i棵子树，选了j门课程的最大学分。

```c++
//#include<bits/stdc++.h>
#include <iostream>
#include <algorithm>
#include<vector>
using namespace std;
int n, m, s[305], k, dp[305][305], visit[305];
vector<int>lesson[6003];
void dfs(int rt) {
    if (visit[rt]) return;
    visit[rt] = 1;
    dp[rt][1] = s[rt];//初始化
    for (auto it : lesson[rt]) {
        dfs(it);
        for (int i = m + 1; i >= 0; i--) {//倒序，保证dp[rt][i-j]中选取的结点只包含it子树前面子树的结点。（参考01背包）
            for (int j = 0; j < i; j++) {//j<i,因为i-j!=0，保证rt结点一定被选上
                dp[rt][i] = max(dp[rt][i], dp[it][j] + dp[rt][i - j]);
            }
        }
    }
}
int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; i++) {
        cin >> k >> s[i];
        lesson[k].push_back(i);
    }
    dfs(0);
    cout << dp[0][m + 1] << "\n";//以0为根，要选m+1个点，所得到的最大学分
}
```

### 悬线法求最大子矩阵

```c++
//动态规划求最优子矩阵用悬线法，悬线法就是从左边上边推到这个点的状态，然后更新这个点的状态然后处理。
//代码是求最大01相间矩阵的
#include<bits/stdc++.h>
using namespace std;  
const int maxn=2010;
int ma[maxn][maxn];
int l[maxn][maxn],r[maxn][maxn],up[maxn][maxn];
int main(){
    int n,m;
    cin>>n>>m;
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            int tmp;
            cin>>tmp;
            ma[i][j]=tmp;
        }
    }
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            l[i][j]=j;
            r[i][j]=j;
            up[i][j]=1;
        }
    }
    //先处理每一行的最长长度，然后处理每一列的，也可以先处理每一列的。
    for(int i=1;i<=n;i++){
        for(int j=2;j<=m;j++){
            if(ma[i][j]!=ma[i][j-1]){
                l[i][j]=l[i][j-1];
            }
        }
        for(int j=m-1;j>=1;j--){
            if(ma[i][j]!=ma[i][j+1]){
                r[i][j]=r[i][j+1];
            }
        }
    }
    int ans1=0,ans2=0;
    //每一行的最长长度已经算了出来，然后处理每一列的。
    //处理过程中
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            if(i==1) continue;
            if(ma[i][j]!=ma[i-1][j]){    //核心部分
                up[i][j]=up[i-1][j]+1;
                l[i][j]=max(l[i-1][j],l[i][j]);
                r[i][j]=min(r[i][j],r[i-1][j]);
            }
            int len=r[i][j]-l[i][j]+1;
            ans1=max(ans1,min(up[i][j],len));
            ans2=max(ans2,up[i][j]*len);
            cout<<i<<" "<<j<<" "<<len<<endl;
            cout<<ans2<<endl;
        }
    }
    cout<<ans1*ans1<<endl;//最大正方形面积
    cout<<ans2<<endl;//最大矩形面积
    return 0;
}
```

### 归并排序（求逆序对）

求逆序对的常用方法还有树状数组

1. 逆序对个数也是冒泡排序需要的最少交换次数

2. 奇数码问题（8数码是n=3的奇数码）

```c++
void MergeSort(int l, int r) {
	//合并a[l,mid]和a[mid+1,r]
	//a是待排序数组，b是临时数组，cnt是逆序对个数
	if (l >= r)return;
	int mid = (l + r) >> 1;
	MergeSort(l, mid);
	MergeSort(mid + 1, r);
	int i = l, j = mid + 1;
	for (int k = l; k <= r; k++) {
		if (j > r || i <= mid && a[i] <= a[j])b[k] = a[i++];
		else {
			b[k] = a[j++];
			cnt += mid - i + 1;
		}
	}
	for (int k = l; k <= r; k++) a[k] = b[k];
}
```

# 图论

### 树上最近公共祖先（lca）

倍增法

首先选定某点为树根，dfs求每个点的深度depth，点 $ x $ 的第 $ 2^i $ 级祖先 $ f[x,i] $ ，点x到第 $ 2^i $ 级祖先的距离。之后lca函数求x到y的距离，

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
struct edge{
	int u,v,w;
}e[50004];
struct ed{
	int v,w; 
};
vector<ed>graph[10004];
int fa[10004],f[10004][35],res[10004][35],dep[10004];
set<int>st;
int find(int x){
	if (fa[x]==x)return fa[x];
	else return fa[x]=find(fa[x]);
}
void Union(int x,int y){
	int fx=find(x),fy=find(y);
	if (fx==fy)return;
	else fa[fx]=fy;
}
bool cmp(edge a,edge b){
	return a.w>b.w;
}
void dfs(int now,int father){
	f[now][0]=father;
	dep[now]=dep[father]+1;
	for (int i=1;i<=30;i++){
		f[now][i]=f[f[now][i-1]][i-1];
		res[now][i]=min(res[f[now][i-1]][i-1],res[now][i-1]);
	}
	for (vector<ed>::iterator it=graph[now].begin();it!=graph[now].end();it++){
		int v=(*it).v,w=(*it).w;
		if (v==father)continue;
		f[v][0]=now;
		res[v][0]=w;
		dfs(v,now);
	}
}
int lca(int x,int y){
	if (dep[x]<dep[y])swap(x,y);
	int cha=dep[x]-dep[y];
	int i=0,ans=0x3f3f3f3f;
	while (cha){
		if (cha&1){
			ans=min(ans,res[x][i]);
			x=f[x][i];
		}
		cha>>=1;
		i++;
	}
	if (x==y)return ans;
	for (i=30;i>=0;i--){
		if (f[x][i]==f[y][i])continue;
		ans=min(ans,min(res[x][i],res[y][i]));
		x=f[x][i];
		y=f[y][i];
	}
	ans=min(ans,min(res[x][0],res[y][0]));
	return ans;
}
signed main (){
	ios::sync_with_stdio(false),cin.tie(0);
//	freopen("P1396_4.in","r",stdin);
//	freopen("P1396_4.out","w",stdout);
	int n,m,x,y,z,q;
	cin>>n>>m;
	for (int i=1;i<=n;i++)fa[i]=i;
	for (int i=1;i<=m;i++){
		cin>>e[i].u>>e[i].v>>e[i].w;
	}
	sort(e+1,e+m+1,cmp);
	//最大生成树 
	for (int i=1;i<=m;i++){
		if (find(e[i].u)==find(e[i].v))continue;
		else{
			Union(e[i].u,e[i].v);
			graph[e[i].u].push_back({e[i].v,e[i].w});
			graph[e[i].v].push_back({e[i].u,e[i].w});
		}
	}
	//倍增法求lca 
	memset(res,0x3f,sizeof(res));
	for (int i=1;i<=n;i++){
		st.insert(find(i));
	}
	for (set<int>::iterator it=st.begin();it!=st.end();it++)dfs(*it,0);
	cin>>q;
	while (q--){
		cin>>x>>y;
		if (find(x)!=find(y))cout<<"-1\n";
		else cout<<lca(x,y)<<"\n";	
	}
}
```

tarjan算法



树链剖分

```c++
#include <bits/stdc++.h>
using namespace std;
const int maxn=500015;
struct edge{
	int v,fail;
}e[maxn*2];
int p[maxn],eid;
void init(){
	memset(p,-1,sizeof p);
	eid=0;
}
void insert(int u,int v){
	e[eid].v=v;
	e[eid].fail=p[u];
	p[u]=eid++;
}
int sz[maxn],dep[maxn],top[maxn],hson[maxn],fa[maxn];
void dfs1(int u){
	sz[u]=1;
	dep[u]=dep[fa[u]]+1;
	for(int i=p[u];~i;i=e[i].fail){
		int v=e[i].v;
		if(v!=fa[u]){
			fa[v]=u;
			dfs1(v);
			sz[u]+=sz[v];
			if(sz[v]>sz[hson[u]]){
				hson[u]=v;
			} 
		}
	} 
}
void dfs2(int u,int tp){
	top[u]=tp;
	if(hson[u]){
		dfs2(hson[u],tp);
	}
	for(int i=p[u];~i;i=e[i].fail){
		int v=e[i].v;
		if(v!=fa[u] && v!=hson[u]){
			dfs2(v,v);
		}
	}
}
int lca(int a,int b){
	while(top[a]!=top[b]){
		if(dep[top[a]]>dep[top[b]])swap(a,b);
		b=fa[top[b]];
	}
	return dep[a]>dep[b]?b:a;
}
int main(){
	init();
	int n,m,s;
	scanf("%d%d%d",&n,&m,&s); 
	for(int i=1;i<n;i++){
		int a,b;
		scanf("%d%d",&a,&b);
		insert(a,b);
		insert(b,a);
	} 
	dfs1(s);
	dfs2(s,s);
	for(int i=1;i<=m;i++){
		int a,b;
		scanf("%d%d",&a,&b);
		printf("%d\n",lca(a,b));
	}
	return 0;
}

```

### 强连通分量（tarjan算法）

我们可以将一张图的每个强连通分量都缩成一个点。然后这张图会变成一个 DAG，可以进行拓扑排序以及更多其他操作。

```c++
//时间复杂度o(n+m)
#define N 100005
vector<int> edge[N];
int stk[N],instk[N],top=0;//stk:栈数组；instk[i]:i是否在栈中；top:栈顶指针
int dfn[N],low[N],tot=0;//dfn:时间戳；low:追溯值；tot:当前访问的点的总数
int scc[N],siz[N],cnt=0;//scc[i]:点i在哪个强连通分量中；siz[i]:强连通分量i的大小；cnt:强连通分量的个数
int indeg[N];
void tarjan(int x){
    //入x时，盖戳，入栈
	dfn[x]=low[x]=++tot;
	stk[++top]=x;
	instk[x]=1;
	for (vector<int>::iterator it=edge[x].begin();it!=edge[x].end();it++){
		int y=*it;
		if (!dfn[y]){//若y未访问
			tarjan(y);
			low[x]=min(low[x],low[y]);//回x时更新low
		}else if (instk[y]){//若y已访问且在栈中
			low[x]=min(low[x],dfn[y]);//更新low
		}
	}
    //离x时更新scc
	if (dfn[x]==low[x]){//若x是scc的根
		int k;
		cnt++;
		do{
			k=stk[top--];
			instk[k]=0;
			scc[k]=cnt;//scc编号
			siz[cnt]++;//scc大小
		}while (x!=k);
	}
}
```

### 链式前向星

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long 
#define N 200005
#define M 200005
int head[N],nxt[M],to[M],val[M],tot;
void add(int u,int v,int w){
	nxt[++tot]=head[u];	// 当前边的后继
	head[u]=tot;		// 起点 u 的第一条边
	to[tot]=v;			// 当前边的终点
	val[tot]=w;
}
signed main(){
	ios::sync_with_stdio(false),cin.tie(0);
	int n,m,u,v,w;
	cin>>n>>m;
	// head[u] 和 cnt 的初始值都为 0
	tot=0;
	memset(head,0,sizeof(int)*(n+1));
	// 建图
	for (int i=1;i<=m;i++){
		cin>>u>>v>>w;
		add(u,v,w);
	}
	// 遍历u的出边
	for (int i=head[u];i;i=nxt[i]){
		int v=to[i];
	}
}
```

### Floyd

```c++
memset(dis,0x3f,sizeof(dis));
for (int i=1;i<=m;i++){
    cin>>u>>v>>w;
    dis[u][v]=w;
}
for (int k = 1; k <= n; k++) {	
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j]);
        }
    }
    // dis[k][i][j] = min(dis[k][i][j], dis[k-1][i][k] + dis[k-1][k][j]);
    // dis[k][i][j]表示除i和j外只经过结点1->k的情况下最短路的长度
    // 第0维可以省略
}
```

### Dijkstra

```c++
struct edge{
	int v,w;
};
struct path{
	int dis,u;
	bool operator>(const path& a)const{return dis>a.dis;}
};
vector<edge>graph[1005];
priority_queue<path,vector<path>,greater<path> >q;
int dis[1005],vis[1005];
void dij(int x){
	memset(dis,0x3f,sizeof(dis));
	memset(vis,0,sizeof(vis));
	dis[x]=0;
	q.push({0,x});
	while(!q.empty()){
		int u=q.top().u;
		q.pop();
		if (vis[u])continue;
		vis[u]=1;
		for (vector<edge>::iterator it=graph[u].begin();it!=graph[u].end();it++){
			int v=(*it).v,w=(*it).w;
			if (vis[v])continue;
			if (dis[v]>dis[u]+w){
				dis[v]=dis[u]+w;
				q.push({dis[v],v});
			}
		}
	}
}
```

### SPFA

```c++
queue<int>q;
void spfa(int s){
	memset(dis,0x3f,sizeof(int)*(n+1));
	q.push(s);
	dis[s]=0;
	vis[s]=1;
	while (!q.empty()){
		int u=q.front();
		q.pop();
		vis[u]=0;
		for (int i=head[u];i;i=nxt[i]){
			int v=to[i],w=val[i];
			if (dis[v]>dis[u]+w){
				dis[v]=dis[u]+w;
				if (!vis[v]){
					q.push(v);
					vis[v]=1;
				}
			}
		}
	}
}
```

### Kruskal 

```c++
const int maxn=100005;
int fa[maxn];
int head[maxn],to[maxn],nxt[maxn],val[maxn],tot=0;
struct edge{
	int u,v,w;
}e[maxn];
int find(int x){
	if (fa[x]==x)return x;
	else return fa[x]=find(fa[x]);
}
void Union(int x,int y){
	int fx=find(x),fy=find(y);
	if (fx==fy)reutrn;
	fa[fx]=fy;
}
bool cmp(edge a,edge b){
	return a.w<b.w;
}
void add(int u,int v,int w){
	nxt[++tot]=head[u];
	head[u]=tot;
	to[tot]=v;
	val[tot]=w;
}
int Kruskal(int s){
	for (int i=1;i<=n;i++)fa[i]=i;
	sort(e+1,e+1+m,cmp);
	int ans=0;
	for (int i=1;i<=m;i++){
		int u=e[i].u,v=e[i].v,w=e[i].w;
		if (find(u)==find(v))continue;
		Union(u,v);
		ans+=w;
		add(u,v,w);
		add(v,u,w); 
	}
	return ans;
```

### Prim

```c++
void prim(){
	memset(dis,0x3f,sizeof(int)*(n+1));
	dis[1]=0; // 以1为树根 
	for (int i=head[1];i;i=nxt[i]){ // 更新1能到达的点的最小距离
		int v=to[i],w=val[i];
		dis[v]=min(dis[v],w); // min可防止重边
	}
	int cnt=0,now=1;
	while (++cnt<=n-1){ // 只需循环n-1次,找n-1条边 
		int mn=0x3f3f3f3f;
		vis[now]=1;
		for (int i=1;i<=n;i++){ // 寻找离now最近的点
			if(!vis[i]&&mn>dis[i]){
				mn=dis[i];
				now=i;
			}
		}
		ans+=mn; // 最短的边加入答案
		for (int i=head[now];i;i=nxt[i]){ //用新的now更新距离
			int v=to[i],w=val[i];
			if (dis[v]>w&&!vis[v])dis[v]=w;
		}
	}
}
```

### 拓扑排序

```c++
int toposort() {
	int cnt = 0;
	for (int i = 1; i <= n; i++) {
		if (indegree[i] == 0) 
			q.push(i);	
	}
	while (!q.empty()) {
		int t = q.front();
		q.pop();
		cnt++;
		topo[cnt]=t;
		for (vector<int>::iterator it = graph[t].begin(); it != graph[t].end(); it++) {
			indegree[*it]--;
			if (indegree[*it] == 0)
				q.push(*it);
		}
	}
	if (cnt == n)
		return 1;
	else
		return 0;
}
```

### BFS

求经过的结点数最少的情况下的最短路径

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
vector<pair<int, int> > graph[100005];
queue<int>q;
int vis[100005], cost[100005];
int n, m, u, v, w;
void bfs() {
    //bfs求经过结点数最少的路径
    //vis[i]表示1到i最少经过的结点数
    //cost[i]表示1到i经过的结点数最少的情况下的最短路径
	q.push(1);
	vis[1] = 1;
	while (!q.empty()) {
		int now = q.front();
		q.pop();
		for (auto it : graph[now]) {
			if (!vis[it.first]) {
				q.push(it.first);
				vis[it.first] = vis[now] + 1;
				cost[it.first] = cost[now] + it.second;
			}
			else if (vis[it.first] == vis[now] + 1) {
				cost[it.first] = min(cost[it.first], cost[now] + it.second);
			}
		}
	}
}
signed main() {
	ios::sync_with_stdio(false), cin.tie(0);
	cin >> n >> m;
	for (int i = 1; i <= m; i++) {
		cin >> u >> v >> w;
		graph[u].push_back(make_pair(v, w));
		graph[v].push_back(make_pair(u, w));
	}
	bfs();
	cout << vis[n] << " " << cost[n] << "\n";
}
```

### 二分图最大匹配

从前一个和谐的班级，有  $ n_l $  个是男生，有 $ n_r $  个是女生。编号分别为 $ 1, \dots, n_l $  和 $ 1, \dots, n_r $ 。有若干个这样的条件：第 $ v $  个男生和第 $ u $  个女生愿意结为配偶。请问这个班级里最多产生多少对配偶？

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
struct edge {
    int nxt,to;
}e[250005];
int cnt=1,head[505],vis[505],match[505],res[505];
void add(int u, int v) {//链式前向星存图
    e[cnt] = edge{ head[u],v };
    head[u] = cnt++;
}
int dfs(int u) {
    for (int i = head[u]; i; i = e[i].nxt) {
        int v = e[i].to;//女生
        if (vis[v]) continue;
        vis[v] = 1;
        if (match[v] == 0 || dfs(match[v])) {//女生未匹配或匹配的男生可以让出女生
            match[v] = u;
            return 1;
        }
    }
    return 0;
}
signed main() {
    ios::sync_with_stdio(false), cin.tie(0);
    int nl, nr, m, u, v;
    cin >> nl >> nr >> m;
    for (int i = 1; i <= m; i++) {
        cin >> u >> v;
        add(u, v);
    }
    int ans = 0;
    for (int i = 1; i <= nl; i++) {
        memset(vis, 0, sizeof(vis));//vis初始化为0
        if (dfs(i))ans++;
    }
    cout << ans << "\n";
}
```

# 树

### 树的直径

两次dfs：首先从任意节点  $ y $  开始进行第一次 DFS，到达距离其最远的节点，记为  $ z $ ，然后再从  $ z $  开始做第二次 DFS，到达距离  $ z $  最远的节点，记为  $ z' $ ，则  $ \delta(z,z') $  即为树的直径。

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
vector<int> graph[200005];
int d[200005],c,pre[200005];
void dfs(int u,int fa){
	pre[u]=fa;
	for (vector<int>::iterator it=graph[u].begin();it!=graph[u].end();it++){
		int v=*it;
		if (v==fa)continue;
		d[v]=d[u]+1;
		if (d[v]>d[c])c=v;
		dfs(v,u);
	}
}
signed main(){
	ios::sync_with_stdio(false),cin.tie(0);
	int n,u,v;
	cin>>n;
	for (int i=1;i<=n-1;i++){
		cin>>u>>v;
		graph[u].push_back(v);
		graph[v].push_back(u);
	} 
	dfs(1,0);
	d[c]=0;
	dfs(c,0);
	cout<<d[c]<<"\n";
	int cnt=0;
	for (int i=c;cnt<=d[c];i=pre[i]){
		cnt++;
		cout<<i<<" ";
	}
}
```

### 霍夫曼树

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
struct vex {
	double weight;
	int id;
	bool friend operator>(vex a, vex b) {
		return a.weight > b.weight;
	}
};

struct node {
	double weight;
	int id;
	node* lchild, * rchild;
}nod[300];
priority_queue<vex, vector<vex>, greater<vex> >q;
int ans[100], cnt = 0;
double cost = 0;
void dfs(int rt) {
	if (nod[rt].lchild == NULL && nod[rt].rchild == NULL) {
		cout << "第" << nod[rt].id << "个指令(使用频率为"<<nod[rt].weight<<")的编码为：";
		for (int i = 1; i <= cnt; i++)cout << ans[i];
		cout << "\n";
		cost += cnt * nod[rt].weight;
		return;
	}
	if (nod[rt].lchild != NULL) {
		ans[++cnt] = 0;
		dfs(nod[rt].lchild->id);
		cnt--;
	}
	if (nod[rt].rchild != NULL) {
		ans[++cnt] = 1;
		dfs(nod[rt].rchild->id);
		cnt--;
	}
}

signed main() {
	ios::sync_with_stdio(false), cin.tie(0);
	int tot = 78;
	for (int i = 1; i <= 10; i++) {
		q.push({ 0.049,i });
		nod[i].weight = 0.049;
		nod[i].id = i;
		nod[i].lchild = nod[i].rchild = NULL;
	}
	for (int i = 11; i <= 28; i++) {
		q.push({ 0.02,i });
		nod[i].weight = 0.02;
		nod[i].id = i;
		nod[i].lchild = nod[i].rchild = NULL;
	}
	for (int i = 29; i <= 78; i++) {
		q.push({ 0.003,i });
		nod[i].weight = 0.003;
		nod[i].id = i;
		nod[i].lchild = nod[i].rchild = NULL;
	}
	while (q.size() >= 2) {
		tot++;
		nod[tot].lchild = nod + q.top().id; q.pop();
		nod[tot].rchild = nod + q.top().id; q.pop();
		nod[tot].weight = nod[tot].lchild->weight + nod[tot].rchild->weight;
		nod[tot].id = tot;
		q.push({ nod[tot].weight,tot });
	}
	dfs(q.top().id);
	cout << "平均码长为：" << cost << "\n";
}
```

### Trie字典树

```c++
int trie[SIZE][26],tot=1,cnt[SIZE];//tot是trie树上节点个数
void insert(string str) {
	int len = str.length(), p = 1;
	for (int i = 0; i < len; i++) {
		int ch = str[i] - 'a';
		if (!trie[p][ch]) trie[p][ch] = ++tot;//结点中存储指针域
		p = trie[p][ch];//p指向子树
		cnt[p]++;//cnt表示满足的字符串个数
	}
}
int search(string str) {
	int len = str.length(), p = 1;
	for (int i = 0; i < len; i++) {
		int ch = str[i] - 'a';
		if (!trie[p][ch])return 0;
		p = trie[p][ch];
	}
	return cnt[p];
}
```

# 字符串

### 最长回文字串manacher

```c++
string s, str;          //s为原字符串，str为添加字符后的字符串
int P[N];               //保存每个字符的回文半径
void add() {
    str+='^';
    for (int i = 0; i < s.size(); i++) {
        str += '#';
        str += s[i];
    }
    str+='#';
    str+='@';
}
void manacher() {
    int R = 0, mid = 0;
    for (int i = 1; i < str.size() - 1; i++) {
        P[i] = R > i ? min(P[2 * mid - i], R - i) : 1;	//进行三种情况的判断
        while (str[i + P[i]] == str[i - P[i]]) P[i]++;	//中心拓展
        if (i + P[i] > R) {                           	//如果当前回文串已经覆盖到了原先没有覆盖到的地方，则更新标记
            R = i + P[i];
            mid = i;
        }
    }
```

### 后缀数组

 $ sa[i] $ 表示将所有后缀排序后第i小的后缀的编号，也是所说的后缀数组；

 $ rk[i] $ 表示后缀i的排名，是重要的辅助数组；

这两个数组满足性质： $ sa[rk[i]]=rk[sa[i]]=i $ 。

 $ height[i] $ 表示第i名的后缀与它前一名的后缀的最长公共前缀,即 $ height[i]=lcp(sa[i],sa[i-1]) $ 

引理： $ height[rk[i]] >= height[rk[i - 1]] - 1 $ 

LCP（最长公共前缀）：  $ lcp(i, j) $  表示后缀  $ i $  和后缀  $ j $  的最长公共前缀（的长度）。

两子串最长公共前缀： $ lcp(sa[i], sa[j]) = \min\{height[i + 1..j]\} $  （转换为RMQ区间最小值问题）

不同子串的数目： $ \frac{ n(n + 1) }{2} - \sum\limits_{ i = 2 } ^ nheight[i] $ 

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
const int N = 200010;
string s;
// key1[i] = rk[id[i]]（作为基数排序的第一关键字数组）
int sa[N], rk[N], oldrk[N << 1], id[N], key1[N], cnt[N],height[N];
bool cmp(int x, int y, int w) {
    return oldrk[x] == oldrk[y] && oldrk[x + w] == oldrk[y + w];
}
void get_sa(int m, int n) {
    //sa[i]表示将所有后缀排序后第i小的后缀的编号
    int i, j, p = 0;
    for (i = 1; i <= n; ++i) ++cnt[rk[i] = s[i]];
    for (i = 1; i <= m; ++i) cnt[i] += cnt[i - 1];
    for (i = n; i >= 1; --i) sa[cnt[rk[i]]--] = i;
    for (j = 1;; j <<= 1, m = p) {
        for (p = 0, i = n; i > n - j; --i) id[++p] = i;
        for (i = 1; i <= n; ++i)  if (sa[i] > j) id[++p] = sa[i] - j;
        memset(cnt, 0, sizeof(cnt));
        for (i = 1; i <= n; ++i) ++cnt[key1[i] = rk[id[i]]];
        for (i = 1; i <= m; ++i) cnt[i] += cnt[i - 1];
        for (i = n; i >= 1; --i) sa[cnt[key1[i]]--] = id[i];
        memcpy(oldrk + 1, rk + 1, n * sizeof(int));
        for (p = 0, i = 1; i <= n; ++i) rk[sa[i]] = cmp(sa[i], sa[i - 1], j) ? p : ++p;
        if (p == n) {
            for (i = 1; i <= n; ++i) sa[rk[i]] = i;
            break;
        }
    }
}
void get_height(int n) {
    //height[i]表示第i名的后缀与它前一名的后缀的最长公共前缀,即height[i]=lcp(sa[i],sa[i-1])
    //引理：height[rk[i]] >= height[rk[i - 1]] - 1
    for (int i = 1, k = 0; i <= n; ++i) {
        if (rk[i] == 0) continue;
        if (k) --k;
        while (s[i + k] == s[sa[rk[i] - 1] + k]) ++k;
        height[rk[i]] = k;
    }
}
signed main() {
    int n;
    cin >> n >> s;
    s += s;
    string temp = s;
    reverse(temp.begin(), temp.end());
    s = " " + s + "#" + temp;
    int len = s.length() - 1;
    get_sa(127, len);
    get_height(len);
    int ans = 0;
    //求字符串s中长度小于等于n的不同子串个数
    for (int i = 1; i <= len; i++)ans += min(sa[i] ,n) - min(height[i + 1], n);
    ans -= n * (n + 1) / 2;
    cout << ans << "\n";
}
```

### KMP算法

```c++
int n;
vector<int> pi(n);
void next(string s) {
	int n = (int)s.length();
	for (int i = 1; i < n; i++) {
		int j = pi[i - 1];
		while (j > 0 && s[i] != s[j]) j = pi[j - 1];
		if (s[i] == s[j]) j++;
		pi[i] = j;
	}
}
int KMP(string s, string t, int pos) {
	int i = pos, j = 1, len1 = s.length(), len2 = t.length();
	while (i <= len1 && j <= len2) {
		if (j == 0 || s[i] == t[j]) {
			++i;
			++j;
		}
		else  j = pi[j];
	}
	if (j > len2)  return  i - len2;
	else return 0;
}
```

# 数学

> 组合数可以通过杨辉三角来递推计算

 $ \sum_{i=0}^n\binom{n-i}{i}=F_{n+1}\tag{12} $  （其中 F 是斐波那契数列）

### 埃式筛

时间复杂度是  $ O(n\log\log n) $ 

```c++
int Eratosthenes(int n) {
  int p = 0;
  for (int i = 0; i <= n; ++i) is_prime[i] = 1;
  is_prime[0] = is_prime[1] = 0;
  for (int i = 2; i <= n; ++i) {
    if (is_prime[i]) {
      prime[p++] = i;
      for (int j = i * i; j <= n; j += i) is_prime[j] = 0;
    }
  }
  return p;
}
```

### 拓展欧几里得

常用于求  $ ax+by=\gcd(a,b) $  的一组可行解。

```c++
int exgcd(int a, int b, int& x, int& y) {
	if (b == 0) {
		x = 1; y = 0;
		return a;
	}
	int d=exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}
```

### 欧拉函数

欧拉函数即  $ \varphi(n) $ ，表示的是小于等于  $ n $  和  $ n $  互质的数的个数。

```c++
int euler_phi(int n) {
  int ans = n;
  for (int i = 2; i * i <= n; i++)
    if (n % i == 0) {
      ans = ans / i * (i - 1);
      while (n % i == 0) n /= i;
    }
  if (n > 1) ans = ans / n * (n - 1);
  return ans;
}
```

### 同余方程

求关于 $  x $ 的同余方程  $  a x \equiv 1 \pmod {b} $  的最小正整数解。

```c++
signed main() {
	ios::sync_with_stdio(false), cin.tie(0);
	int a, b, x, y;
	cin >> a >> b;
	exgcd(a, b, x, y);
	cout << (x % b + b) % b << "\n";
}
```

### 汉明权重

汉明权重是一串符号中不同于（定义在其所使用的字符集上的）零符号（zero-symbol）的个数。对于一个二进制数，它的汉明权重就等于它1的个数（即 `popcount`)。

在 [状压 DP](https://oi-wiki.org/dp/state/) 中，按照 popcount 递增的顺序枚举有时可以避免重复枚举状态。这是构造汉明权重递增的排列的一大作用。

```c++
for (int i=(1<<r)-1,t;i<=(1<<n)-1;t=i+(i&-i),i=i?(t|((((t&-t)/(i&-i))>>1)-1)):(i+1)){
    //汉明权重为r,最多有n位的所以排列
}
```

### 线性基

求在一个序列中，取若干个数，使得它们的异或和最大/最小

```c++
int p[55];
void insert(int x){
	for (int i=50;i>=0;i--){
		if (!(x>>i))continue;
		if (!p[i]){
			p[i]=x;
			break;
		}
		x^=p[i];
	}
}
for (int i=1;i<=n;i++){
		cin>>x;
		insert(x);
	} 
	for (int i=50;i>=0;i--)ans=max(ans,ans^p[i]);
```

### Miller-Rabin

```c++
int Quick_Multiply_Mod(int a, int b, int m){
	int ans = 0, temp = a;
	while (b){
		if (b & 1)ans = (ans + temp) % m;
		temp = (temp + temp) % m;
		b >>= 1;
	}
	return ans;
}
int ksm(int di,int mi,int mod){
	int ans=1;
	while (mi){
		if (mi&1)ans=Quick_Multiply_Mod(ans,di,mod);
		di=Quick_Multiply_Mod(di,di,mod);
		mi>>=1;
	}
	return ans;
}
bool Miller_Rabin(int n){//Miller-Rabing算法
	if (n == 2)//2是素数
		return true;
	if (n < 2 || n % 2 == 0)//0，1和偶数不是素数
		return false;
	//把n-1写成2的k次方*t的形式
	int k=0, t=n-1;
	while (!(t&1))//如果t不是奇数，就执行。相当于t%2
	{
		k++;
		t >>= 1;//t向右移动一位，相当于t/2
	}
	//进行20轮测试，增加可靠性
	for (int i = 0; i <=20; i++){
		int a = rand() % (n - 1) + 1;//选取底数a，1<=a<=n-1
		int b = ksm(a, t, n);
		int y;
		for (int i = 0; i < k; i++){
			y = Quick_Multiply_Mod(b, b, n);
			if (y == 1 && b != 1 && b != n - 1)
				return false;
			b = y;
		}
		if (y != 1)
			return false;
	}
	return true;
}
```

### 矩阵快速幂

```c++
struct matrix{
	int m[4][4],h,l;
}mat,ans,temp;
matrix operator *(const matrix a,const matrix b){
	matrix c;
	for (int i=1;i<=a.h;i++){
		for (int j=1;j<=b.l;j++){
			c.m[i][j]=0;
			for (int k=1;k<=a.l;k++){
				c.m[i][j]=(c.m[i][j]+a.m[i][k]*b.m[k][j]%mod)%mod;
			}
		}
	}
	c.h=a.h;
	c.l=b.l;
	return c;
} 
matrix ksm(matrix di,int mi){
	matrix res;
	for (int i=1;i<=3;i++){
		for (int j=1;j<=3;j++)
			res.m[i][j]=0;
	}
	for (int i=1;i<=3;i++)res.m[i][i]=1;
	res.h=res.l=3;
	while (mi){
		if (mi%2==0){
			mi/=2;
			di=di*di;
		}else{
			res=res*di;
			mi--;
		}
	}
	return res;
}
```

### **阶乘逆元法**

```c++
//递推
inv[1] = 1;
for(int i = 2; i <= n; ++ i) inv[i] = (p - p / i) * inv[p % i] % p;   
//阶乘
g[0]=1;
for (int i=1;i<=n;i++)g[i]=(g[i-1]*i)%p;
f[n]=ksm(g[n],p-2);
for (int i=n-1;i>=1;i--)f[i]=(f[i+1]*(i+1))%p;    
for (int i=1;i<=n;i++)inv[i]=f[i]*g[i-1]%p;
for (int i=1;i<=n;i++)cout<<inv[i]<<'\n';
```

### 博弈论



# 计算几何

![img](https://img-blog.csdn.net/20180524193743628?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dzeF85OTk5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### 扫描线

离散化+线段树

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
struct Line {//扫描线
	int x,y1,y2,flag;
}l[200005]; 
struct tree{//线段树
	int l,r,len,cnt;
}t[1600005];
int y[200005];
bool cmp(Line a,Line b){
	return a.x<b.x;
}
void build(int rt,int l,int r){//建树
    //离散化线段树与普通线段树不同
    //1.叶子节点r=l+1
    //2.左右子树分别是[l,mid]和[mid,r]
	t[rt].l=y[l];
	t[rt].r=y[r];
	if (r==l+1)return;
	int mid=(l+r)>>1;
	build(rt<<1,l,mid);
	build(rt<<1|1,mid,r);
}
void pushup(int rt){
    //如果这一段被覆盖过，高度就等于区间长度
	if (t[rt].cnt)t[rt].len=t[rt].r-t[rt].l;
    //否则
	else t[rt].len=t[rt<<1].len+t[rt<<1|1].len;
}
void update(int rt,int a,int b,int c){
	if (a>=t[rt].r||b<=t[rt].l)return;
	else if (a<=t[rt].l&&b>=t[rt].r)t[rt].cnt+=c;
	else {
		update(rt<<1,a,b,c);
		update(rt<<1|1,a,b,c);
	}
	pushup(rt);
}
signed main(){
	ios::sync_with_stdio(false),cin.tie(0);
	int n,x1,y1,x2,y2,ans=0;
	cin>>n;
	for (int i=1;i<=n;i++){
		cin>>x1>>y1>>x2>>y2;
		y[i]=y1;
		y[i+n]=y2;
		l[i]={x1,y1,y2,1};
		l[i+n]={x2,y1,y2,-1};
	}
	n*=2;
	sort(y+1,y+n+1);
	sort(l+1,l+1+n,cmp);
	build(1,1,n);
	for (int i=1;i<n;i++){
		update(1,l[i].y1,l[i].y2,l[i].flag);
		ans+=(l[i+1].x-l[i].x)*t[1].len;
	}
	cout<<ans<<"\n";
} 
```
