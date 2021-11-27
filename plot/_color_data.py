from collections import OrderedDict

# from colorcet
class AttrODict(OrderedDict):
    """Ordered dictionary with attribute access (e.g. for tab completion)"""
    def __dir__(self): return self.keys()
    def __delattr__(self, name): del self[name]
    def __getattr__(self, name):
        return self[name] if not name.startswith('_') else super(AttrODict, self).__getattr__(name)
    def __setattr__(self, name, value):
        if (name.startswith('_')): return super(AttrODict, self).__setattr__(name, value)
        self[name] = value


# CN colors
CN = ['C{}'.format(i) for i in range(10)]
# matplotlib tabelau colors, same as CN
mpl_10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# matplotlib tabelau colors, plus their lightened colors
mpl_20 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#3397dc', '#ff993e', '#3fca3f', '#df5152', '#a985ca',
          '#ad7165', '#e992ce', '#999999', '#dbdc3c', '#35d8e9']

# https://vega.github.io/vega/docs/schemes/
vega_20 = ['#1F77B4', '#AEC7E8', '#FF7F0E', '#FFBB78', '#2CA02C',
           '#98DF8A', '#D62728', '#FF9896', '#9467BD', '#C5B0D5',
           '#8C564B', '#C49C94', '#E377C2', '#F7B6D2', '#7F7F7F',
           '#C7C7C7', '#BCBD22', '#DBDB8D', '#17BECF', '#17BECF']

R_8 = ['#ffffff', '#df536b', '#61d04f', '#2297e6', '#28e2e5', '#cd0bbc', '#f5c710', '#9e9e9e']

mathematica97 = ['#5e81b5', '#e19c24', '#8fb032', '#eb6235', '#8778b3',
                 '#c56e1a', '#5d9ec7', '#ffbf00', '#a5609d', '#929600',
                 '#e95536', '#6685d9', '#f89f13', '#bc5b80', '#47b66d']

mathematica98 = ['#4a969c', '#e28617', '#9d6095', '#85a818', '#d15739',
                 '#6f7bb8', '#e9ac03', '#af5b71', '#38a77e', '#dd6f22',
                 '#8468b8', '#c2aa00', '#b8575c', '#48909f', '#dd8516']

# colors from seaborn
deep = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
        "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

muted = ["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4",
         "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"]

pastel = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
          "#DEBB9B", "#FAB0E4", "#CFCFCF", "#FFFEA3", "#B9F2F0"]

bright = ["#023EFF", "#FF7C00", "#1AC938", "#E8000B", "#8B2BE2",
          "#9F4800", "#F14CC1", "#A3A3A3", "#FFC400", "#00D7FF"]

dark = ["#001C7F", "#B1400D", "#12711C", "#8C0800", "#591E71",
        "#592F0D", "#A23582", "#3C3C3C", "#B8850A", "#006374"]

colorblind = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
              "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"]

simple = ['#003e7c', '#ef7b00', '#146341', '#a8003b', '#2d3742',
          '#cee4cc', '#dfca97', '#413c39', '#e4e1ca', '#d90b36']


cc = AttrODict()
cc['CN'] = CN
cc['mpl_10'] = mpl_10
cc['mpl_20'] = mpl_20
cc['vega_20'] = vega_20
cc['math97'] = mathematica97
cc['math98'] = mathematica98
cc['deep'] = deep
cc['muted'] = muted
cc['pastel'] = pastel
cc['bright'] = bright
cc['dark'] = dark
cc['colorblind'] = colorblind
cc['simple'] = simple