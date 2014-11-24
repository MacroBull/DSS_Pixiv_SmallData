# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 22:49:34 2014
Project	:Python-Project
Version	:0.0.1
@author	:macrobull (http://github.com/macrobull)

"""
import time

def loadDB(fn):
	def spliter(s):
		r = s.strip().split('","')
		r[0] = r[0][1:]
		r[-1] = r[-1][:-1]
		return r

	s=open(fn).readlines()
	r = {}
	keys = spliter(s[0])
	for i in keys:
		r[i]=[]

	for l in s[1:]:
		for i, p in enumerate(spliter(l)):
			try:
				t = int(p)
				r[keys[i]].append(t)
			except ValueError:
				r[keys[i]].append(p)

	from collections import namedtuple

	dbClass=namedtuple('DB',' '.join(r.keys()))
	return dbClass(**r)

def parseDate(s):
	import datetime
	try:
		r = [time.mktime(datetime.datetime.strptime(p,
			"%d/%m/%Y").timetuple()) for p in s]
	except ValueError:
		r = [time.mktime(datetime.datetime.strptime(p,
			"%m/%d/%Y").timetuple()) for p in s]

	return r, s

def parseTime(s):
	import datetime
	try:
		r = [time.mktime(datetime.datetime.strptime(p,
			"%d/%m/%Y %H:%M").timetuple()) for p in s]
	except ValueError:
		r = [time.mktime(datetime.datetime.strptime(p,
			"%m/%d/%Y %H:%M").timetuple()) for p in s]

	return r, s

def xCompress(l):
	r = [l[0]]
	for t in l:
		if t != r[-1]:
			r.append(t)
	return r

def mInt(l):
	return [int(t) for t in l]

tl = loadDB('TopList.csv')
main = loadDB('Illust_main.csv')
tag = loadDB('Illust_tag.csv')
tags = loadDB('Illust_tags.csv')
view = loadDB('Illust_view.csv')


from pylab import *
from macrobull.misc import mplTheme

#mplTheme('none')

td, tPost = parseDate(tl.Date)
tm, tComp = parseTime(main.Time)
#tl.Rank = mInt(tl.Rank)


ill = {}



## RankPlot
'''
subplot(111, title = u'排位变化')
for i in range(len(tl.url)):
	if tl.url[i] not in ill: ill[tl.url[i]] = []
	ill[tl.url[i]].append( (td[i], tl.Rank[i]) )

for i in ill:
	x = []
	y = []
	for xy in ill[i]:
		x.append(xy[0])
		y.append(xy[1])
	plot(x, y, 'o-')

ylim(0, 50)
xlim(np.array(td).min(), np.array(td).max())
xlabel(u'日期')
ylabel(u'排位')

xticks(xCompress(td), xCompress(tPost), rotation = 40)
plt.gca().invert_yaxis()
'''

## RankCombo
'''
subplot(111, title = u'上榜时长')

for i in range(len(tl.url)):
	if tl.url[i] not in ill: ill[tl.url[i]] = 0
	ill[tl.url[i]]+=1

hist(ill.values())
xlabel(u'天数')
ylabel(u'作品数')
'''

## RankRamp
'''
subplot(111, title = u'在榜上升')

for i in range(len(tl.url)):
	if tl.url[i] not in ill: ill[tl.url[i]] = []
	ill[tl.url[i]].append(tl.Rank[i])

hist([int(t[0]) - int(t[-1]) for t in ill.values() if len(t) > 1], 45 )
xlabel(u'排位变化')
ylabel(u'作品数')
'''

## RankOut
'''
subplot(111, title = u'上榜速度')

for i in range(len(tl.url)):
	ill[tl.url[i]] = td[i]

for i in range(len(main.url)):
	if main.url[i] not in ill: ill[main.url[i]] = 0
	if (ill[main.url[i]]>=0) : ill[main.url[i]] -= tm[i]
r = [ill[t] / 3600. + 24 for t in ill if -4e5<ill[t]<4e5]
hist(r, 23)
xlim(0, 48)
xticks(range(0,48 + 4,4))
xlabel(u'上榜时间-创作时间/小时')
ylabel(u'作品数')
'''

# View&Score
'''
from scipy.stats import kendalltau
t,p = kendalltau(view.View, view.Score)
subplot(111, title = u'观看数与总评分' + r'$(\tau={}$'.format(t) + r'$,\overline{p}=$' + r'${})$'.format(p))

hist(view.View, 30 , label = '观看数')
hist(view.Score, 30 , label = '评分')


legend(loc='best')
xlabel(u'数值')
ylabel(u'作品数')
'''

# View&Score vs Rank
#'''
xs = {}
ys = {}

for i in range(len(tl.url)):
	if (time.time() - td[i] ):
		xs[tl.url[i]] = tl.Rank[i]

for i in range(len(view.url)):
	ys[view.url[i]] = view.View[i]

xp = array([xs[x] for x in xs if x in ys])
yp = array([ys[y] for y in ys if y in xs])


from scipy.stats import kendalltau
import seaborn as sns

mplTheme('macrobull')

p = sns.jointplot(xp, yp, kind="hex", stat_func=kendalltau, color="#63AC82")
p.fig.suptitle(u'排位 - 查看数')
xlim(0,50)
ylim(0,15e4)

for i in range(len(view.url)):
	ys[view.url[i]] = view.Score[i]

yp = array([ys[y] for y in ys if y in xs])

p = sns.jointplot(xp, yp, kind="hex", stat_func=kendalltau, color="#7A66AC")
p.fig.suptitle(u'排位 - 评分')
xlim(0,50)
ylim(0,15e4)
#'''

# Tag Dist
'''
utags = {}

for i in range(len(tag.url)):
	flag = True
	for ex in [u'入り']:
		if ex in str(tag.tag[i]):
			flag = False
	if flag:
		if tag.tag[i] not in utags: utags[tag.tag[i]] = 0
		utags[tag.tag[i]] += 1

import operator
topSize = 30
stags = sorted(utags.items(), key=operator.itemgetter(1), reverse = True)
otags = stags[:topSize]

subplot(111, title=u'标签人气')
bar(range(topSize), zip(*otags)[1], color = '#4BAC8F')
xticks(range(topSize), zip(*otags)[0], rotation = -40, ha='left')
ylabel(u'出现次数')
'''


# Tag Change
'''
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
otags = otags[:10]
r = []
for (t,c) in otags:
	x = []
	for i in range(len(tags.url)):
		if t in tags.tags[i]:
			x.append(tm[i])
	r.append(x)

subplot(111, title=u'标签人气时间分布')
hist(r, 36, stacked = True, label = zip(*otags)[0], histtype='stepfilled')

xticks(xCompress(td), xCompress(tPost), rotation = 30)
legend(loc = 'best')
xlim(1.4137e9, 1.4154e9)
ylabel(u'出现次数')
'''

'''
import networkx as nx

g = nx.Graph(name = u'标签聚类')

i = 0
candidates = zip(*stags)[0][:120]

while i<len(tag.url):
	j = i+1
	while (j<len(tag.url)) and (tag.url[j] == tag.url[i]): j+=1
	for k in range(i+1,j):
		if str(tag.tag[k]) in candidates:
			for l in range(i,k):
				if str(tag.tag[l]) in candidates:
					try:
						r = g[tag.tag[k]][tag.tag[l]]
						g[tag.tag[k]][tag.tag[l]]['weight'] += 1
					except KeyError:
						g.add_edge(tag.tag[k], tag.tag[l], weight = 1)
	i = j

pos=nx.spring_layout(g, dim = 2, k = 1.5/sqrt(len(candidates)))

ct = pos[stags[0][0]]
cl = [sqrt((ct[0]-pos[n][0])**2+(ct[1]-pos[n][1])**2) for n in g.node]

sl = [(log(utags[t])+1)*200 for t in g.node]

nx.draw_networkx_edges(g,pos, width = 0.8, alpha=0.4, style = 'solid')
nx.draw_networkx_nodes(g,pos,
	node_color = cl, cmap = cm.coolwarm_r,
#	node_color = '#5B92C5',
	node_size = sl,
	node_shape = 'h',
	linewidths = 1.5,
	alpha = 0.8)
nx.draw_networkx_labels(g, pos,
	font_family = 'Droid Sans Fallback',
	alpha = 0.7)
'''

#illustrator
'''
ilst = {}
for i in range(len(tl.url)):
	if tl.Illustrator[i] not in ilst: ilst[tl.Illustrator[i]] = 0
	ilst[tl.Illustrator[i]] +=1

import operator
topSize = 30
silst = sorted(ilst.items(), key=operator.itemgetter(1), reverse = True)
oilst = silst[:topSize]

subplot(111, title=u'人气作者')
bar(range(topSize), zip(*oilst)[1], color = '#4BAC8F')
xticks(range(topSize), zip(*oilst)[0], rotation = -40, ha='left')
ylabel(u'上榜次数')
'''

show()
