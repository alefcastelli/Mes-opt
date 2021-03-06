# Importing pyomo module
from pyomo.environ import *
from optimization_test import *
import numpy as np
import pandas as pd
from matplotlib.pyplot import *

## Parameters
t=24*7*3 # parameter to set the dimension of the problem
# Demands profiles
EE_demand   = typ_profiles[0:t, 0]
Heat_demand = typ_profiles[0:t, 1]
Cold_demand = typ_profiles[0:t, 2]
PV_output = typ_profiles[0:t, 3] # kW/m2

# Fuels prices [€/kWh]
Fuels = {
    'NG': 0.0258,
    'Diesel':  1
    }

# Electricity prices [€/kWh] # può essere usato anche un profilo <--
#El_price = 0.3*np.ones(t)
#El_sold_price=0.1261*np.ones(t)
#El_purch_price=0.1577*np.ones(t)
El_sold_price=typ_profiles[0:t, 4]
El_purch_price=typ_profiles[0:t, 5]


# Variation penalty cost [€]
var_pen = 8


# Heat producers parameters. Defined as a dictionary
# Fuel: price related to the fuel.
# goods: goods produced by the machines
# m and q parameters: machines goods curve coefficients (straight lines, el=electricity, th=thermal)
# min/max_In: energy input range
# RU/RDlim: are the ramp up/down limits related to the machines (maximum load variation between two time steps)
# RUSU/RDSD: ramp limits related to start-up (SU) and shut-down (SD) of the machines (max load variation between
# time steps when machine start from switched-off condition)
# Dissipable_Heat: True if Heat can be dissipated into atmosphere. NOTE: just for heat producing machines
# Internal: True if it can consume internal goods (e.g. compressor chiller can use energy produced by other machiens)
# NOTE: machines that can consume goods produced by other machiens have fuel cost null. There is a constraint that defines the
# machine input as the sum of the bought electricity and the produced one

# --> in futuro aggiungere anche il tipo di macchina e il numero di priorità
Machines_parameters = {
    'Boiler': { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat'],  'RUlim': 1000, 'RDlim': 10000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 2, 'minDT': 0, 'OM': 3, 'SUcost':0.0503555, 'Dissipable_Heat': False, 'External Consumer':True, 'Internal Consumer': False, 'K1':{'El':0, 'Heat':0.976, 'Cold':0}, 'K2':{'El':0, 'Heat':-0.032, 'Cold':0}, 'K3':{'El':0, 'Heat':4.338, 'Cold':0}, 'KIn_min':0.25, 'KIn_max':1, 'XD_min':100, 'XD_max':50000,'x_design_pws':[0,100, 1000, 10000, 50000], 'Cinv_pws':[0,8, 69, 554, 2387], 'StartLoss':{'El':0, 'Heat':0.2, 'Cold':0},'DeltaT_loss':0.25},
    #'ICE':    { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat', 'El'], 'RUlim': 10000, 'RDlim': 10000, 'RUSU': 10000, 'RDSD': 10000, 'minUT': 6, 'minDT': 0, 'OM': 18, 'SUcost':0.076959, 'Dissipable_Heat': True, 'External Consumer':True, 'Internal Consumer': False, 'K1':{'El':0.49, 'Heat':0.439, 'Cold':0}, 'K2':{'El':-0.017, 'Heat':-0.005, 'Cold':0}, 'K3':{'El':-128.8, 'Heat':108.18, 'Cold':0}, 'KIn_min':0.54, 'KIn_max':1, 'XD_min':1328, 'XD_max':38692, 'x_design_pws':[0,1328, 13783, 26237, 38692], 'Cinv_pws':[0,247, 2296, 4242, 6145], 'StartLoss':{'El':0.2, 'Heat':0.2, 'Cold':0}, 'DeltaT_loss':0.25},
    'HP':      { 'In': 'El', 'fuel cost':           0, 'goods':       ['Heat'], 'RUlim': 1000, 'RDlim': 1000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 0, 'minDT': 0, 'OM':  3, 'SUcost':0.1186441, 'Dissipable_Heat': False, 'External Consumer':False, 'Internal Consumer':  True , 'K1':{'El':0, 'Heat':3.59, 'Cold':0}, 'K2':{'El':0, 'Heat':-0.08, 'Cold':0}, 'K3':{'El':0, 'Heat':0, 'Cold':0}, 'KIn_min':0.13, 'KIn_max':1, 'XD_min':100, 'XD_max':10000, 'x_design_pws':[0,100, 500, 2000, 10000], 'Cinv_pws':[0,254, 778, 2039, 6239], 'StartLoss':{'El':0, 'Heat':0, 'Cold':0}, 'DeltaT_loss':0.25},
    #'CC':      { 'In': 'El', 'fuel cost':           0, 'goods':       ['Cold'],  'RUlim': 1000, 'RDlim': 1000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 2, 'minDT': 0, 'OM':  3, 'SUcost':0.0,  'Dissipable_Heat': False, 'External Consumer':False, 'Internal Consumer':  True, 'K1':{'El':0, 'Heat':0, 'Cold':11.10}, 'K2':{'El':0, 'Heat':0, 'Cold':-0.324}, 'K3':{'El':0, 'Heat':0, 'Cold':0}, 'KIn_min':0.13, 'KIn_max':1, 'XD_min':165, 'XD_max':680, 'x_design_pws':[0,165, 337, 508, 680], 'Cinv_pws':[0,248, 428, 587, 733], 'StartLoss':{'El':0, 'Heat':0, 'Cold':0}, 'DeltaT_loss':0.25}
}

Res_parameters = {
   'PV': {'In': 'SunIrradiance',  'goods': ['El'],  'OM': 10, 'InvCost':300 , 'available area': 10, 'El':PV_output, 'Heat':np.zeros(t), 'Cold':np.zeros(t)} # maintenance €/m2
}
Storage_parameters = {
     'TES': { 'good': 'Heat', 'minC': 0, 'maxC': 1274, 'Init%': 0, 'eta_ch': 0.95, 'eta_disch': 0.95, 'eta_sd': 0.1, 'PmaxIn': 5000, 'PmaxOut': 5000, 'FinCval': 0.0001, 'OMxTP': 0.0001, 'InvCost': 500, 'XD_min':0, 'XD_max':10000, 'x_stor_pws':[0,1,2000,5000,10000], 'Cinv_stor_pws':[0,5, 626, 1131, 1770] },
    'BESS': {'good': 'El', 'minC': 0, 'maxC': 1000, 'Init%': 0, 'eta_ch': 0.99, 'eta_disch': 0.99, 'eta_sd': 0.01, 'PmaxIn': 500, 'PmaxOut': 500, 'FinCval': 0.0001, 'OMxTP': 0.0001, 'InvCost': 500, 'XD_min':0, 'XD_max':1000, 'x_stor_pws':[0, 100, 250, 500, 1000], 'Cinv_stor_pws':[0, 50, 125, 250, 500]},
     'CES': {'good': 'Cold', 'minC': 0, 'maxC': 1000, 'Init%': 0, 'eta_ch': 0.95, 'eta_disch': 0.95, 'eta_sd': 0.1, 'PmaxIn': 500, 'PmaxOut': 500, 'FinCval': 0.0001, 'OMxTP': 0.0001, 'InvCost': 500, 'XD_min': 0, 'XD_max': 1000, 'x_stor_pws':[0, 100, 250, 500, 1000], 'Cinv_stor_pws':[0, 50, 125, 250, 500]}
}

Networks_parameters = {
    "Network": {
        "El": {'sold_price': El_sold_price, 'purch_price':El_purch_price},
        "Heat":{'sold_price': np.zeros(t), 'purch_price':np.ones(t)*1000},  # paramters set so that heat network is not considered in the solution
        #"Cold":{'sold_price': np.zeros(t), 'purch_price':np.ones(t)*1000}   # paramters set so that cold network is not considered in the solution
    }
}

Loads_parameters = {
    'El': {'demand': EE_demand},
    'Heat': {'demand': Heat_demand},
    'Cold': {'demand': Cold_demand}
}

# Time values
# Number of time intervals
T =t
Dt=1 # time step lenght in hours
######################################################################################################################################################################
# Optimization Model #################################################################################################################################################

# Create model as ConcreteModel
model = ConcreteModel()

## SETS
# Set of all time intervals - in this case must be 504
model.times = RangeSet( 0, T - 1 )

# Set of all the machines
list_Machine_ExtCons=[i for i in Machines_parameters.keys() if Machines_parameters[i]["External Consumer"] ]
list_Machine_IntCons=[i for i in Machines_parameters.keys() if Machines_parameters[i]["Internal Consumer"] ]
list_Machine_diss=[i for i in Machines_parameters.keys() if Machines_parameters[i]["Dissipable_Heat"] ]

model.Machines = Set( initialize = Machines_parameters.keys() )
model.Machines_ExtCons = Set(within=model.Machines, initialize=list_Machine_ExtCons)
model.Machines_IntCons = Set(within=model.Machines, initialize=list_Machine_IntCons)
model.Machines_diss= Set( within=model.Machines, initialize=list_Machine_diss)

# Set for RES or Non-controllable/dispatchable units
model.Machines_Res = Set( initialize= Res_parameters.keys())

# Set for storage
model.Storages = Set ( initialize = Storage_parameters.keys() )

# Set for networks
model.Networks = Set (initialize=Networks_parameters.keys())

# Set for MES goods
model.Goods = Set (initialize=["El", "Heat"]) #"Cold"])

# Set for machines slots
n_slots=3 # defined a priori
model.Slots = RangeSet(0, n_slots-1)

# Set of vertex for the convex hull formulation (one-degree of freedom)
model.v = Set(initialize=['Min', 'Max'])

# Creating the dictionary for the K_In paremters
v={}
for i in Machines_parameters.keys():
    v[i, 'Min']=Machines_parameters[i]['KIn_min']
    v[i, 'Max']=Machines_parameters[i]['KIn_max']
model.K_In=Param(model.Machines, model.v, initialize=v)


# Creating dictionaries for the step piecewise cost function
x_pws={}
Cinv_pws={}
for m in Machines_parameters.keys():
    for s in model.Slots:
        x_pws[(m,s)]=Machines_parameters[m]['x_design_pws']
        Cinv_pws[(m,s)]=Machines_parameters[m]['Cinv_pws']

x_stor_pws={}
Cinv_stor_pws={}
for es in Storage_parameters.keys():
        x_stor_pws[es]=Storage_parameters[es]['x_stor_pws']
        Cinv_stor_pws[es]=Storage_parameters[es]['Cinv_stor_pws']

# Initializing the index matrix for cycling constraints
deltaT_cluster=24*7 # hours contained in a cluster --> one week
index_cluster=np.zeros(options['n_clusters'])
for i in range(options['n_clusters']):
    index_cluster[i]=deltaT_cluster*i

## VARIABLES

## Binary variables
# Variable to define if technology t is installed in  site s
model.z_design = Var (model.Machines, model.Slots, domain=Binary)
model.z_design_stor = Var (model.Storages, domain=Binary)
# On/off variable
model.z = Var( model.Machines, model.Slots, model.times, domain = Binary )
# Delta on/off
model.delta_on = Var (model.Machines, model.Slots, model.times, domain=Binary)
model.delta_off = Var (model.Machines, model.Slots, model.times, domain=Binary)
# Binary variable to take into account if storage is charging (1) or discharging (0)
model.c = Var ( model.Storages, model.times, domain=Binary)
# Binary variable to take into account if electricity is sold (1) or purchased (0)
model.s = Var ( model.Networks, model.Goods, model.times, domain=Binary)

## Continuous Variables
# Variable to define the size of unit
model.x_design= Var (model.Machines, model.Slots, domain=NonNegativeReals)
model.x_design_stor= Var(model.Storages, domain=NonNegativeReals)
# Variable to define the area of PV technology installed
model.ResArea = Var (model.Machines_Res, domain=NonNegativeReals)

# Convex hull operation control variable associated to vertex v of the performance map of machine m in site s at time t
model.beta = Var (model.Machines, model.Slots, model.times, model.v-['Min'], domain=NonNegativeReals)
# Linearization variable of the product x_D[m,s] * z[m,s,t]
model.psi = Var (model.Machines, model.Slots, model.times, domain=NonNegativeReals)
# Linearization variable of the product x_D[m,s] * delta_on[m,s,t]
model.eps_on = Var (model.Machines, model.Slots, model.times, domain=NonNegativeReals)

# Variable for input (fuel, el, heat, cold, etc.) [kW]
model.In = Var( model.Machines, model.Slots, model.times, domain=NonNegativeReals)

# Variable for machine output (el, heat, cold, etc.) [kW]
model.Out = Var( model.Machines, model.Slots, model.Goods, model.times, domain=NonNegativeReals)
model.Out_diss = Var( model.Machines_diss, model.Slots, model.Goods, model.times, domain=NonNegativeReals)
model.Out_us = Var( model.Machines_diss, model.Slots, model.Goods, model.times, domain=NonNegativeReals)

model.Out_Res = Var (model.Machines_Res, model.Goods, model.times, domain=NonNegativeReals)

# Energy purch (negative) /sold (positive) to the network n at time t
model.Net_exch = Var (model.Networks, model.Goods, model.times, domain=Reals)

# Storage variables (state of charge (SOC), charge/discharge)
model.SOC = Var (model.Storages, model.times, domain=NonNegativeReals)
model.store_net = Var(model.Storages, model.times, domain=Reals)
model.store_char = Var(model.Storages, model.times, domain=NonNegativeReals)
model.store_disch = Var(model.Storages, model.times, domain=NonNegativeReals)

# Investment cost associated to machine m in slot s
model.Cinv=Var(model.Machines, model.Slots, domain=NonNegativeReals)
model.Cinv_stor=Var(model.Storages, domain=NonNegativeReals)
# Net revenues from the exchange of good with the external networks (ex. electricity --> positive if sold, negative if purchased)
model.Net_rev= Var (model.Networks, model.Goods, model.times, domain=Reals)


## OBJECTIVE FUNCTION
# Interest rate for the TAC calculation
CCR=0.15

def ObjFun( model ):
    return (  ( sum(model.Cinv[m, s] for m in model.Machines for s in model.Slots) + sum(
            model.Cinv_stor[es] for es in model.Storages )  )*1000 \
            + sum(model.ResArea[r]*Res_parameters[r]['InvCost'] for r in model.Machines_Res)  )*CCR + sum (
            (           - sum(model.Net_rev[n,g,t] for n in model.Networks for g in model.Goods) + sum(model.In[m, s , t]*Machines_parameters[m]['fuel cost']
                        for m in model.Machines_ExtCons for s in model.Slots) + sum(model.z[m, s, t]*Machines_parameters[m]['OM']
                        for m in model.Machines for s in model.Slots) + sum(model.ResArea[r]*Res_parameters[r]['OM'] for r in model.Machines_Res) + sum(
                        model.delta_on[m, s, t]*Machines_parameters[m]['SUcost']*Machines_parameters[m]['fuel cost']
                        for m in model.Machines_ExtCons for s in model.Slots)     )
            for t in model.times )

model.obj = Objective(
    rule = ObjFun,
    sense = minimize
    )

## CONSTRAINTS

## Investment Constraints
# The same site cannot be assigned to more than one machine: N_max machine per site
N_max=1
def sites_perMachine_rule( model, s):
    return sum(model.z_design[m,s] for m in (model.Machines)) <= N_max
model.sites_perMachine_constr=Constraint(model.Slots, rule=sites_perMachine_rule)
# Simmetry breaking constraint on the site filling with machines
model.constr=ConstraintList()
for k,m in enumerate(Machines_parameters.keys()):
        for s in range(n_slots-1):
            if s >= k:
                model.constr.add( model.z_design[m, s+1] <= model.z_design[m, s] )

# Link between z design and z operational: only installed unit can be operated
def z_link_rule( model, m, s, t):
    return model.z[m, s, t]  <= model.z_design[m,s]
model.z_link_constr=Constraint(model.Machines, model.Slots, model.times, rule=z_link_rule)


# Design bound constraints
def XD_max_rule(model, m, s):
    return model.x_design[m, s] <= model.z_design[m, s]*Machines_parameters[m]['XD_max']
def XD_min_rule(model, m, s):
    return model.x_design[m, s] >= model.z_design[m, s]*Machines_parameters[m]['XD_min']
def XD_max_stor(model, es):
    return model.x_design_stor[es] <= model.z_design_stor[es]*Storage_parameters[es]['XD_max']
def XD_min_stor(model, es):
    return model.x_design_stor[es] >= model.z_design_stor[es]*Storage_parameters[es]['XD_min']
def Available_area_rule(model, r):
    return model.ResArea[r] <= Res_parameters[r]["available area"]

model.XD_max_constr=Constraint(model.Machines, model.Slots, rule=XD_max_rule)
model.XD_min_constr=Constraint(model.Machines, model.Slots, rule=XD_min_rule)
model.XD_max_stor_constr=Constraint(model.Storages, rule=XD_max_stor)
model.XD_min_stor_constr=Constraint(model.Storages, rule=XD_min_stor)
model.Available_area_constr=Constraint (model.Machines_Res, rule=Available_area_rule)

# Constraint to enforce the piecewise interpolation
model.Pws_constr = Piecewise(model.Machines, model.Slots, model.Cinv, model.x_design,
                      pw_pts=x_pws,
                      pw_constr_type='EQ',
                      f_rule=Cinv_pws,
                      pw_repn='SOS2',
                      unbounded_domain_var=True)
model.Pws_stor_constr = Piecewise(model.Storages, model.Cinv_stor, model.x_design_stor,
                      pw_pts=x_stor_pws,
                      pw_constr_type='EQ',
                      f_rule=Cinv_stor_pws,
                      pw_repn='SOS2',
                      unbounded_domain_var=True)

####

# Machine part load performance expressed as a convex combination of its operating vertexes
# Linearization of the bilinear term psi[m,s,t]=x_D[m,s]*z[m,s,t] and beta[m,s,t,v]=x_D[m,s]*alpha[m,s,t,v]
def psiBeta_rule(model, m, s, t):
    return sum( model.beta[m,s,t,v] for v in model.v-['Min']) <= model.psi[m,s,t]
def psi_rule1(model, m, s, t):
    return model.psi[m,s,t] <= model.z[m,s,t] * Machines_parameters[m]['XD_max']
def psi_rule2(model, m, s, t):
    return model.psi[m,s,t] <= model.x_design[m,s]
def psi_rule3(model, m, s, t):
    return model.psi[m,s,t] >= model.x_design[m,s] - (1-model.z[m,s,t]) * Machines_parameters[m]['XD_max']
model.psi_constr1=Constraint(model.Machines, model.Slots, model.times, rule=psiBeta_rule)
model.psi_constr2=Constraint(model.Machines, model.Slots, model.times, rule=psi_rule1)
model.psi_constr3=Constraint(model.Machines, model.Slots, model.times, rule=psi_rule2)
model.psi_constr4=Constraint(model.Machines, model.Slots, model.times, rule=psi_rule3)

# Linearization of the bilinear term eps_on[m,s,t]=x_D[m,s]*delta_on[m,s,t]
def eps_on_rule1(model, m, s, t):
    return model.eps_on[m,s,t] <= model.delta_on[m,s,t] * Machines_parameters[m]['XD_max']
def eps_on_rule2(model, m, s, t):
    return model.eps_on[m,s,t] <= model.x_design[m,s]
def eps_on_rule3(model, m, s, t):
    return model.eps_on[m,s,t] >= model.x_design[m,s] - (1-model.delta_on[m,s,t]) * Machines_parameters[m]['XD_max']
model.eps_on_constr2=Constraint(model.Machines, model.Slots, model.times, rule=eps_on_rule1)
model.eps_on_constr3=Constraint(model.Machines, model.Slots, model.times, rule=eps_on_rule2)
model.eps_on_constr4=Constraint(model.Machines, model.Slots, model.times, rule=eps_on_rule3)

# Min/Max energy input constraint
def machines_minIn( model, m, s, t ):
    return  model.In[m, s, t] >= model.psi[m,s,t]* Machines_parameters[m]['KIn_min']
model.machine_constr_minIn=Constraint(model.Machines, model.Slots, model.times, rule=machines_minIn)
def machines_maxIn( model, m, s, t ):
    return  model.In[m, s, t] <= model.psi[m,s,t] * Machines_parameters[m]['KIn_max']
model.machine_constr_maxIn=Constraint(model.Machines, model.Slots, model.times, rule=machines_maxIn)

# Definition of variables In on the basis of Beta (convex hull/combination)
def In_rule(model, m, s, t):
    return model.In[m, s, t] == model.K_In[m,'Min']*model.psi[m,s,t] + sum( model.beta[m,s,t,v]*(model.K_In[m,v]-model.K_In[m, 'Min']) for v in model.v-['Min'])
model.In_constr=Constraint(model.Machines, model.Slots, model.times, rule=In_rule)

# Machine ouput definition
def out_func(model, m, s, g, t):
    return model.psi[m,s,t]*(model.K_In[m,'Min']*Machines_parameters[m]['K1'][g] +Machines_parameters[m]['K2'][g]) + sum(
            Machines_parameters[m]['K1'][g]*(model.K_In[m,v]-model.K_In[m,'Min'])*model.beta[m,s,t,v] for v in model.v-['Min']) \
           + Machines_parameters[m]['K3'][g]*model.z[m,s,t] - Machines_parameters[m]['StartLoss'][g]*model.eps_on[m,s,t]*Machines_parameters[m]['DeltaT_loss']
def Output_rule(model, m, s, g, t):
    return model.Out[m, s, g, t] == out_func(model, m, s, g, t)
model.Output_constr=Constraint(model.Machines, model.Slots, model.Goods, model.times, rule=Output_rule)

# Res and Non Dispatchable sources production modelling
def out_func_Res(model, r, g, t):
    return Res_parameters[r][g][t]*model.ResArea[r]
def OutRes_rule(model, r, g, t):
    return model.Out_Res[r, g, t] == out_func_Res(model, r, g, t)
model.OutRes_constr=Constraint(model.Machines_Res, model.Goods, model.times, rule=OutRes_rule)

# Out Useful constraint: out useful if generated but not dissipated
# A constraint that the system uses to define model.Out_diss each time step
def diss_rule( model, m, s, g, t ):
    return model.Out_us[m, s, g, t] == model.Out[m, s, g, t] - model.Out_diss[m, s, g, t]
model.diss_constr = Constraint(model.Machines_diss, model.Slots, model.Goods, model.times, rule = diss_rule)


# Storage operation-investment constraints
def Stor_SOCSize_link_rule(model, es, t):
    return model.SOC[es,t] <= model.x_design_stor[es]
model.Stor_levelSize_constr=Constraint(model.Storages, model.times, rule=Stor_SOCSize_link_rule)

# Storage power limits constraint
def stor_powerIn_rule ( model, s, t):
    return (model.store_char[s,t]) <= Storage_parameters[s]['PmaxIn']*model.c[s,t]  # kWh= kW*h
model.stor_powerIn_constr = Constraint(model.Storages, model.times, rule=stor_powerIn_rule)
def stor_powerOut_rule ( model, s, t):
    return (model.store_disch[s,t]) <= Storage_parameters[s]['PmaxOut']*(1-model.c[s,t])   # kWh/h = kW
model.stor_powerOut_constr = Constraint(model.Storages, model.times, rule=stor_powerOut_rule)

# Storage net energy exchange defintion
def stor_net_rule(model, s, t):
    return model.store_net[s, t] == model.store_disch[s, t]*Storage_parameters[s]['eta_disch'] - model.store_char[s, t]/Storage_parameters[s]['eta_ch']
model.stor_net_constr = Constraint(model.Storages, model.times, rule=stor_net_rule)

# Constraint to fix the storage initial level
def Stor_init_rule(model, es):
    return model.SOC[es,0] == Storage_parameters[es]["Init%"]*model.x_design_stor[es]
model.Stor_init_constr=Constraint(model.Storages, rule=Stor_init_rule)

# Storage level link
def stor_link_rule(model, s):
    return model.SOC[s,0]==model.SOC[s,T-1]
model.stor_link_constr = Constraint(model.Storages, rule=stor_link_rule)


# Define the relevant parameters to limit the network operation
for n in Networks_parameters.keys():
    for g in model.Goods:
        Networks_parameters[n][g]['M1']= (max(Loads_parameters[g]['demand']) + sum(Machines_parameters[m]['XD_max']*n_slots for m in model.Machines_IntCons if g in Machines_parameters[m]['In']) )
        Networks_parameters[n][g]['M2']= sum(Machines_parameters[m]['XD_max']*n_slots for m in Machines_parameters.keys() if g in Machines_parameters[m]['goods'])

# Set of constraint to establish wether the electricity is sold or purchased
def Networks_rule1( model, n, g, t):
    return model.Net_exch[n, g, t] >= (model.s[n, g, t]-1) * Networks_parameters[n][g]["M1"]
def Networks_rule2( model, n, g, t):
    return model.Net_exch[n, g, t] <= (model.s[n, g, t]) * Networks_parameters[n][g]["M2"]
model.Networks_constr1 = Constraint(model.Networks, model.Goods, model.times, rule=Networks_rule1)
model.Networks_constr2 = Constraint(model.Networks, model.Goods, model.times, rule=Networks_rule2)

def Network_deactivate_rule( model, n, t):
    return model.Net_exch[n, 'Heat', t]==0
model.Network_deact_constr=Constraint(model.Networks, model.times, rule=Network_deactivate_rule)

# Set of constraints to define El_tot
def Networks_rev_rule1( model, n, g, t):
    return model.Net_rev[n, g, t] <= model.Net_exch[n, g, t]*Networks_parameters[n][g]["sold_price"][t] + (1-model.s[n, g, t]) * Networks_parameters[n][g]["purch_price"][t]*Networks_parameters[n][g]["M1"]
def Networks_rev_rule2( model, n, g, t):
    return model.Net_rev[n, g, t] <= model.Net_exch[n, g, t]*Networks_parameters[n][g]["purch_price"][t] + (model.s[n, g, t]) *Networks_parameters[n][g]["sold_price"][t]* Networks_parameters[n][g]["M2"]
model.Networks_rev_constr1 = Constraint(model.Networks, model.Goods, model.times, rule=Networks_rev_rule1)
model.Networks_rev_constr2 = Constraint(model.Networks, model.Goods, model.times, rule=Networks_rev_rule2)

### Cycling constraints

for i in range(options['n_clusters']):  #iterating over the number of clusters
    if i < options['n_clusters']-1:  # index_cluster[i] represents the first timestep of each clusters
        TT=index_cluster[i+1]-1   # getting the last timestep of the i-th cluster
    else:
        TT=T-1
    for t in range(int(index_cluster[i]), int(TT+1)): # t going from t_start to t_end cluster
        tt=np.arange(t, t+2)  # getting two consecutives timesteps (t, t+1)
        tt[tt>TT]=tt[tt>TT]-TT+index_cluster[i]-1      # cycling constraint --> the end is linked with the cluster start
        # Storage level constraint
        for s in model.Storages:
            model.constr.add( model.SOC[s, tt[1]] == model.SOC[s, tt[0]] * (1 - Storage_parameters[s]['eta_sd']) + (
                            model.store_char[s, tt[1]] - model.store_disch[s, tt[1]]) )
        # Machines constraints
        for m in model.Machines:
            for s in model.Slots:
                # Ramps limit and start up/shut down constraints
                model.constr.add((model.In[m, s, tt[1]] - model.In[m, s, tt[0]]) <= model.z[m,s,tt[0]]*Machines_parameters[m]['RUlim'] + (1-model.z[m,s,tt[0]])*Machines_parameters[m]['RUSU'])
                model.constr.add((model.In[m, s, tt[1]] - model.In[m, s, tt[0]]) >= -model.z[m,s,tt[1]]*Machines_parameters[m]['RDlim'] - (1-model.z[m,s,tt[1]])*Machines_parameters[m]["RDSD"])
                # Delta on constraint
                model.constr.add( model.delta_on[m, s, tt[1]] >= (model.z[m, s, tt[1]]-model.z[m, s, tt[0]]) )
                model.constr.add( model.delta_on[m, s, tt[1]] <= (model.z[m, s, tt[1]] - model.z[m, s, tt[0]] + 1)/2 )
                #Delta off constraint
                model.constr.add( model.delta_off[m, s, tt[1]] >= -(model.z[m, s, tt[1]]-model.z[m, s, tt[0]]) )
                model.constr.add( model.delta_off[m, s, tt[1]] <=  (-model.z[m, s, tt[1]] + model.z[m, s, tt[0]] + 1)/2 )

# Min UT/DT constraints
for i in range(options['n_clusters']):  #iterating over the number of clusters
    if i < options['n_clusters']-1:
        TT=index_cluster[i+1]-1   # defining the last timestep of the i-th cluster
    else:
        TT=T-1
    for m in model.Machines:
        for s in model.Slots:
            for t in range(int(index_cluster[i]), int(TT+1)): # t going from t_start to t_end cluster
                tUT=np.arange(t, t+Machines_parameters[m]['minUT'])  # t_up timesteps definition
                tDT=np.arange(t, t+Machines_parameters[m]['minDT'])  # t_down timesteps definition
                tUT[tUT>TT]=tUT[tUT>TT]-TT+index_cluster[i]-1      # cycling constraint --> the end is linked with the cluster start
                tDT[tDT > TT] = tDT[tDT > TT] - TT + index_cluster[i] - 1
                for tt in tUT:
                    model.constr.add(model.z[m,s,tt]>= model.delta_on[m,s,t])
                for tt in tDT:
                    model.constr.add((1-model.z[m, s, tt]) >= model.delta_off[m, s, t])


# Energy balance constraint rule
def energy_balance_rule( model, g, t ):
    return sum( model.Out[m, s, g, t] for m in model.Machines for s in model.Slots ) - \
           sum(model.Out_diss[m , s, g, t] for m in model.Machines_diss for s in model.Slots) + \
           sum( model.Out_Res[r, g, t] for r in model.Machines_Res) - \
           sum( model.In[m, s, t] for m in model.Machines_IntCons for s in model.Slots if g in Machines_parameters[m]['In']) + \
           sum(model.store_net[es, t] for es in model.Storages if g in Storage_parameters[es]['good']) - \
           sum(model.Net_exch[n, g, t] for n in model.Networks)  == Loads_parameters[g]['demand'][t]

# Energy Balance
model.Energy_balance_constr = Constraint(
    model.Goods,
    model.times,
    rule = energy_balance_rule
    )

## Solve PROBLEM
model.solver=SolverFactory('gurobi')
results = model.solver.solve(model, options={'mipgap':0.01},  tee=True) # tee=True to display solver output in console
results.write() # display results summary in console
#options={'mipgap':0.01},
#model.pprint()

#SolverFactory('gurobi').solve(model).write()
