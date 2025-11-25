# %%
import pickle as pkl
import numpy as np 
import hist
from collections import defaultdict


# %%
prepend = "./inputs/"
filenames = [
    prepend + "pythia_2016_syst.pkl",
    prepend + "pythia_2016APV_syst.pkl",
    prepend + "pythia_2017_syst.pkl",
    prepend + "pythia_2018_syst.pkl",
]

# %%
def group(h: hist.Hist, oldname: str, newname: str, grouping: dict[str, list[str]]):
    hnew = hist.Hist(
        hist.axis.StrCategory(grouping, name=newname),
        *(ax for ax in h.axes if ax.name != oldname),
        storage=h.storage_type,
    )
    for i, indices in enumerate(grouping.values()):
        hnew.view(flow=True)[i] = h[{oldname: indices}][{oldname: sum}].view(flow=True)

    return hnew

# %%
keymap = {
    "2016": "pythia_UL16NanoAODv9",
    "2016APV": "pythia_UL16NanoAODAPVv9",
    "2017": "pythia_UL17NanoAODv9",
    "2018": "pythia_UL18NanoAODv9",
}
htbin_list = ['pythia_HT-100to200', 'pythia_HT-200to400', 'pythia_HT-400to600', 'pythia_HT-1200to2500', 'pythia_HT-2500toInf', 'pythia_HT-800to1200', 'pythia_HT-600to800']

# %%
response_dict = {}
for filename in filenames:
    era = filename.split('_')[1]
    key = keymap[era]
    response_dict.setdefault('u', {})
    response_dict.setdefault('g', {})

    
    grouping = defaultdict(list)
    grouping[key] = htbin_list
    with open(filename, "rb") as f:
        data = pkl.load(f)
        # ensure top-level 'u' and 'g' keys exist
        
        if era == "2016APV" or era == "2016":
            response_dict['u'][key] = data['response_matrix_u'].project('dataset', 'ptgen', 'mgen', 'ptreco', 'mreco', 'systematic')
            response_dict['g'][key] = data['response_matrix_g'].project('dataset', 'ptgen', 'mgen', 'ptreco', 'mreco', 'systematic')
            continue
        h_old = data['response_matrix_u'].project('dataset', 'ptgen', 'mgen', 'ptreco', 'mreco', 'systematic')
        h_new = group(h_old, oldname="dataset", newname="dataset", grouping=dict(grouping))
        response_dict['u'][key] = h_new

        h_old = data['response_matrix_g'].project('dataset', 'ptgen', 'mgen', 'ptreco', 'mreco', 'systematic')
        h_new = group(h_old, oldname="dataset", newname="dataset", grouping=dict(grouping))
        response_dict['g'][key] = h_new
 

# %%
# grouping = defaultdict(list)
# grouping[new_key].append(ds)
# h = group(h, oldname="dataset", newname="dataset", grouping=dict(grouping))

# %%
correlation_dic = {
    'JES_AbsoluteMPFBias': 1,
    'JES_AbsoluteScale': 1,
    'JES_AbsoluteStat': 0,
    'JES_FlavorQCD': 1,
    'JES_Fragmentation': 1,
    'JES_PileUpDataMC': 0.5,
    'JES_PileUpPtBB': 0.5,
    'JES_PileUpPtEC1': 0.5,
    'JES_PileUpPtEC2': 0.5,
    'JES_PileUpPtHF': 0.5,
    'JES_PileUpPtRef': 0.5,
    'JES_RelativeFSR': 0.5,
    'JES_RelativeJEREC1': 0,
    'JES_RelativeJEREC2': 0,
    'JES_RelativeJERHF': 0.5,
    'JES_RelativePtBB': 0.5,
    'JES_RelativePtEC1': 0,
    'JES_RelativePtEC2': 0,
    'JES_RelativePtHF': 0.5,
    'JES_RelativeBal': 0.5,
    'JES_RelativeSample': 0,
    'JES_RelativeStatEC': 0,
    'JES_RelativeStatFSR': 0,
    'JES_RelativeStatHF': 0,
    'JES_SinglePionECAL': 1,
    'JES_SinglePionHCAL': 1,
    'JES_TimePtEta': 0,
    'JER': 0,
}

jes_sys_list = ['JES_AbsoluteMPFBiasUp', 'JES_AbsoluteMPFBiasDown', 'JES_AbsoluteScaleUp', 'JES_AbsoluteScaleDown',
                'JES_AbsoluteStatUp', 'JES_AbsoluteStatDown', 'JES_FlavorQCDUp', 'JES_FlavorQCDDown', 'JES_FragmentationUp',
                'JES_FragmentationDown', 'JES_PileUpDataMCUp', 'JES_PileUpDataMCDown', 'JES_PileUpPtBBUp', 'JES_PileUpPtBBDown',
                'JES_PileUpPtEC1Up', 'JES_PileUpPtEC1Down', 'JES_PileUpPtEC2Up', 'JES_PileUpPtEC2Down', 'JES_PileUpPtHFUp', 'JES_PileUpPtHFDown', 
                'JES_PileUpPtRefUp', 'JES_PileUpPtRefDown', 'JES_RelativeFSRUp', 'JES_RelativeFSRDown', 'JES_RelativeJEREC1Up',
                'JES_RelativeJEREC1Down', 'JES_RelativeJEREC2Up', 'JES_RelativeJEREC2Down', 'JES_RelativeJERHFUp', 'JES_RelativeJERHFDown',
                'JES_RelativePtBBUp', 'JES_RelativePtBBDown', 'JES_RelativePtEC1Up', 'JES_RelativePtEC1Down', 'JES_RelativePtEC2Up', 'JES_RelativePtEC2Down',
                'JES_RelativePtHFUp', 'JES_RelativePtHFDown', 'JES_RelativeBalUp', 'JES_RelativeBalDown', 'JES_RelativeSampleUp', 'JES_RelativeSampleDown', 
                'JES_RelativeStatECUp', 'JES_RelativeStatECDown', 'JES_RelativeStatFSRUp', 'JES_RelativeStatFSRDown', 'JES_RelativeStatHFUp', 'JES_RelativeStatHFDown',
                'JES_SinglePionECALUp', 'JES_SinglePionECALDown', 'JES_SinglePionHCALUp', 'JES_SinglePionHCALDown', 'JES_TimePtEtaUp', 'JES_TimePtEtaDown', 'JERUp', 'JERDown']


non_jes_sys_list = ['nominal', 'puUp', 'puDown', 'elerecoUp', 'elerecoDown',
                    'eleidUp', 'eleidDown', 'eletrigUp', 'eletrigDown', 'murecoUp',
                    'murecoDown', 'muidUp', 'muidDown', 'mutrigUp', 'muisoUp', 'muisoDown','mutrigDown', 'pdfUp',
                    'pdfDown', 'q2Up', 'q2Down', 'l1prefiringUp', 'l1prefiringDown', 
                     'JMRUp', 'JMRDown', 'JMSUp', 'JMSDown']


non_jes_sys_list_up = [sys for sys in non_jes_sys_list if sys[-2:] == 'Up' ]
non_jes_sys_list_down = [sys for sys in non_jes_sys_list if sys[-4:] == 'Down' ]

jes_sys_list_up = [sys for sys in jes_sys_list if sys[-2:] == 'Up' ]
jes_sys_list_down = [sys for sys in jes_sys_list if sys[-4:] == 'Down' ]

# %%
sys_matrix_dic_up = {}
groomed = True
if not groomed:
    response = response_dict['u']
else:
    response = response_dict['g']

for sys in jes_sys_list_up:
    m_nom_2016 = response['pythia_UL16NanoAODv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
    m_nom_2017 = response['pythia_UL17NanoAODv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
    m_nom_2018 = response['pythia_UL18NanoAODv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
    
    m_sys_2016 = response['pythia_UL16NanoAODv9'][..., sys].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
    m_sys_2017 = response['pythia_UL17NanoAODv9'][..., sys].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
    m_sys_2018 = response['pythia_UL18NanoAODv9'][..., sys].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
    
    
    m_var_2016 = m_sys_2016 + m_nom_2017 + m_nom_2018
    m_var_2017 = m_nom_2016 + m_sys_2017 + m_nom_2018
    m_var_2018 = m_nom_2016 + m_nom_2017 + m_sys_2018
    
    
    rho = 0.5 ## correlation factor
    
    sigma_2016 = m_sys_2016 - m_nom_2016
    sigma_2017 = m_sys_2017 - m_nom_2017
    sigma_2018 = m_sys_2018 - m_nom_2018
    
    sigma_corr = rho*sigma_2016 + rho*sigma_2017 + rho*sigma_2018
    
    sigma_uncorr_2016 = (1-rho)*sigma_2016
    sigma_uncorr_2017 = (1-rho)*sigma_2017
    sigma_uncorr_2018 = (1-rho)*sigma_2018
    
    m_nom =  m_nom_2016 + m_nom_2017 + m_nom_2018
    m_corr = m_nom + sigma_corr
    
    m_uncorr_2016 = m_nom + sigma_uncorr_2016
    
    m_uncorr_2017 = m_nom + sigma_uncorr_2017

    m_uncorr_2018 = m_nom + sigma_uncorr_2018

    

    sys_matrix_dic_up[sys+'_corr'] = m_corr
    sys_matrix_dic_up[sys+'_uncorr_2016'] = m_uncorr_2016
    sys_matrix_dic_up[sys+'_uncorr_2017'] = m_uncorr_2017
    sys_matrix_dic_up[sys+'_uncorr_2018'] = m_uncorr_2018
    
non_jes_sys_matrix_dic_up = {}
for sys in non_jes_sys_list_up:
    variation16 = response['pythia_UL16NanoAODv9'][..., sys].project('ptgen','mgen','ptreco','mreco').values() 
    variation17 = response['pythia_UL17NanoAODv9'][..., sys].project('ptgen','mgen','ptreco','mreco').values()
    variation18 = response['pythia_UL18NanoAODv9'][..., sys].project('ptgen','mgen','ptreco','mreco').values()
    variation = variation16 + variation17 + variation18
    sys_matrix_dic_up[sys] = variation
    non_jes_sys_matrix_dic_up[sys] = variation



sys_matrix_dic_down = {}
for sys in jes_sys_list_down:
    m_nom_2016 = response['pythia_UL16NanoAODv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
    m_nom_2017 = response['pythia_UL17NanoAODv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
    m_nom_2018 = response['pythia_UL18NanoAODv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
    
    m_sys_2016 = response['pythia_UL16NanoAODv9'][..., sys].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
    m_sys_2017 = response['pythia_UL17NanoAODv9'][..., sys].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
    m_sys_2018 = response['pythia_UL18NanoAODv9'][..., sys].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
    
    
    m_var_2016 = m_sys_2016 + m_nom_2017 + m_nom_2018
    m_var_2017 = m_nom_2016 + m_sys_2017 + m_nom_2018
    m_var_2018 = m_nom_2016 + m_nom_2017 + m_sys_2018
    
    
    rho = 0.5 ## correlation factor
    
    sigma_2016 = m_sys_2016 - m_nom_2016
    sigma_2017 = m_sys_2017 - m_nom_2017
    sigma_2018 = m_sys_2018 - m_nom_2018
    
    sigma_corr = rho*sigma_2016 + rho*sigma_2017 + rho*sigma_2018
    
    sigma_uncorr_2016 = (1-rho)*sigma_2016
    sigma_uncorr_2017 = (1-rho)*sigma_2017
    sigma_uncorr_2018 = (1-rho)*sigma_2018
    
    m_nom =  m_nom_2016 + m_nom_2017 + m_nom_2018
    m_corr = m_nom + sigma_corr
    
    m_uncorr_2016 = m_nom + sigma_uncorr_2016
    
    m_uncorr_2017 = m_nom + sigma_uncorr_2017

    m_uncorr_2018 = m_nom + sigma_uncorr_2018

    

    sys_matrix_dic_down[sys+'_corr'] = m_corr
    sys_matrix_dic_down[sys+'_uncorr_2016'] = m_uncorr_2016
    sys_matrix_dic_down[sys+'_uncorr_2017'] = m_uncorr_2017
    sys_matrix_dic_down[sys+'_uncorr_2018'] = m_uncorr_2018
    
non_jes_sys_matrix_dic_down = {}
for sys in non_jes_sys_list_down:
    variation16 = response['pythia_UL16NanoAODv9'][..., sys].project('ptgen','mgen','ptreco','mreco').values() 
    variation17 = response['pythia_UL17NanoAODv9'][..., sys].project('ptgen','mgen','ptreco','mreco').values()
    variation18 = response['pythia_UL18NanoAODv9'][..., sys].project('ptgen','mgen','ptreco','mreco').values()
    variation = variation16 + variation17 + variation18
    sys_matrix_dic_down[sys] = variation
    non_jes_sys_matrix_dic_down[sys] = variation
m_nom = m_nom_2016 + m_nom_2017 + m_nom_2018

# sys_matrix_dic['herwigUp'] = resp_matrix_4d_herwig.project('ptgen','mgen','ptreco','mreco').values()
# sys_matrix_dic_down['herwigDown'] = resp_matrix_4d_herwig.project('ptgen','mgen','ptreco','mreco').values()

# %%
import matplotlib.pyplot as plt
for key in sys_matrix_dic_up.keys():
    plt.stairs(sys_matrix_dic_up[key].sum(axis = (0,1,2)), label=key)
for key in sys_matrix_dic_down.keys():
    plt.stairs(sys_matrix_dic_down[key].sum(axis = (0,1,2)), label=key)
plt.yscale('log')

# %%
import matplotlib.pyplot as plt
import hist
import mplhep as hep

mreco_nom = m_nom.sum(axis = (0,1,2))
total_unc_up = np.zeros_like(mreco_nom)
for key in sys_matrix_dic_up.keys():
    mreco_sys = sys_matrix_dic_up[key].sum(axis = (0,1,2))
    delta = mreco_sys - mreco_nom
    total_unc_up += delta**2
total_unc_up = np.sqrt(total_unc_up)
total_unc_down = np.zeros_like(mreco_nom)
for key in sys_matrix_dic_down.keys():
    mreco_sys = sys_matrix_dic_down[key].sum(axis = (0,1,2))
    delta = mreco_sys - mreco_nom
    total_unc_down += delta**2
total_unc_down = np.sqrt(total_unc_down)
bins = np.arange(len(mreco_nom) + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

plt.style.use(hep.style.CMS)  # optional styling
fig, ax = plt.subplots()
hep.histplot(mreco_nom, bins=bins, histtype='band', color='black',yerr=[total_unc_down, total_unc_up], label='Stat. + Syst. Unc.',ax=ax, facecolor = 'cyan', hatch = '', alpha = 1)
hep.histplot(mreco_nom, bins=bins, histtype='band', color='black',yerr=[total_unc_down/2, total_unc_up/2], label='Stat Unc.',ax=ax, facecolor = 'blue', hatch = '', alpha = 1)
#ax.errorbar(bin_centers, mreco_nom, yerr=[total_unc_down, total_unc_up], fmt='none', color='black', label='total uncertainty')
ax.set_yscale('log')
ax.set_xlabel('bin')
ax.set_ylabel('counts')
ax.legend()

# %%
import plotly.graph_objects as go

fig = go.Figure()
for key in sys_matrix_dic_up.keys():
    y_data = sys_matrix_dic_up[key].sum(axis=(0, 1, 2))
    x_data = list(range(len(y_data)))
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', line_shape='hv', name=key))
for key in sys_matrix_dic_down.keys():
    y_data = sys_matrix_dic_down[key].sum(axis=(0, 1, 2))
    x_data = list(range(len(y_data)))
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', line_shape='hv', name=key))
fig.update_layout(yaxis_type="log")
fig.show()

# %%



