# Importing pyomo module
from pyomo.environ import *
from optimization_test import *
import numpy as np
import pandas as pd
from matplotlib.pyplot import *

## Parameters
t=24 # parameter to set the dimension of the problem
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
    'Boiler': { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat'],    'm_th': 0.976, 'q_th': -160.0,  'm_el':   0.0, 'q_el':    0.0, 'min_In': 1250, 'max_In': 5000, 'RUlim': 1000, 'RDlim': 10000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 2, 'minDT': 0, 'OM': 3, 'SUcost':0.0503555, 'Dissipable_Heat': False, 'External Consumer':True, 'Internal Consumer': False, 'K1':{'El':0, 'Heat':0.976, 'Cold':0}, 'K2':{'El':0, 'Heat':-0.032, 'Cold':0}, 'K3':{'El':0, 'Heat':4.338, 'Cold':0}, 'KIn_min':0.25, 'KIn_max':1, 'XD_min':0, 'XD_max':50000 },
    'ICE':    { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat', 'El'], 'm_th': 0.439, 'q_th': -391.82, 'm_el': 0.490, 'q_el': -298.83, 'min_In': 5000, 'max_In': 10000, 'RUlim': 10000, 'RDlim': 10000, 'RUSU': 10000, 'RDSD': 10000, 'minUT': 6, 'minDT': 0, 'OM': 18, 'SUcost':0.076959, 'Dissipable_Heat': True, 'External Consumer':True, 'Internal Consumer': False, 'K1':{'El':0.49, 'Heat':0.439, 'Cold':0}, 'K2':{'El':-0.017, 'Heat':-0.005, 'Cold':0}, 'K3':{'El':0, 'Heat':-108.18, 'Cold':-128.83}, 'KIn_min':0.54, 'KIn_max':1, 'XD_min':0, 'XD_max':38692},
    'HP':      { 'In': 'El', 'fuel cost':           0, 'goods':       ['Heat'],    'm_th': 3.59, 'q_th':    -80.0, 'm_el': 0.000, 'q_el':    0.0, 'min_In':    130, 'max_In': 1000, 'RUlim': 1000, 'RDlim': 1000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 0, 'minDT': 0, 'OM':  3, 'SUcost':0.1186441, 'Dissipable_Heat': False, 'External Consumer':False, 'Internal Consumer':  True , 'K1':{'El':0, 'Heat':3.59, 'Cold':0}, 'K2':{'El':0, 'Heat':-0.08, 'Cold':0}, 'K3':{'El':0, 'Heat':0, 'Cold':0}, 'KIn_min':0.13, 'KIn_max':1, 'XD_min':0, 'XD_max':10000},
    'CC':      { 'In': 'El', 'fuel cost':           0, 'goods':       ['Cold'],    'm_th': 3.500, 'q_th':    0.00, 'm_el': 0.000, 'q_el':    0.0, 'min_In':    0, 'max_In': 2500, 'RUlim': 1000, 'RDlim': 1000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 2, 'minDT': 0, 'OM':  3, 'SUcost':0.0,  'Dissipable_Heat': False, 'External Consumer':False, 'Internal Consumer':  True, 'K1':{'El':0, 'Heat':0, 'Cold':11.10}, 'K2':{'El':0, 'Heat':0, 'Cold':-0.324}, 'K3':{'El':0, 'Heat':0, 'Cold':0}, 'KIn_min':0.13, 'KIn_max':1, 'XD_min':0, 'XD_max':680},
}
Res_parameters = {
   'PV': {'In': 'SunIrradiance',  'goods': ['El'],  'OM': 10, 'InvCost':300 , 'available area': 10, 'El':PV_output, 'Heat':np.zeros(t), 'Cold':np.zeros(t)} # maintenance €/m2
}
Storage_parameters = {
     'TES1': { 'good': 'Heat', 'minC': 0, 'maxC': 1274, 'Init%': 0, 'eta_ch': 0.95, 'eta_disch': 0.95, 'eta_sd': 0.05, 'PmaxIn': 5000, 'PmaxOut': 5000, 'FinCval': 0.0001, 'OMxTP': 0.0001, 'InvCost': 500, 'XD_min':0, 'XD_max':10000 },
     'EES1': {'good': 'El', 'minC': 0, 'maxC': 1000, 'Init%': 0, 'eta_ch': 0.95, 'eta_disch': 0.95, 'eta_sd': 0.1, 'PmaxIn': 500, 'PmaxOut': 500, 'FinCval': 0.0001, 'OMxTP': 0.0001, 'InvCost': 500, 'XD_min':0, 'XD_max':1000} # €/kWh
}

Networks_parameters = {
    "El Grid": { 'good': 'El', 'sold_price': El_sold_price, 'purch_price':El_purch_price}
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
list_stor_el=[i for i in Storage_parameters.keys() if "El" in Storage_parameters[i]['good']]
list_stor_heat=[i for i in Storage_parameters.keys() if "Heat" in Storage_parameters[i]['good']]
list_stor_cold=[i for i in Storage_parameters.keys() if "Cold" in Storage_parameters[i]['good']]

model.Storages = Set ( initialize = Storage_parameters.keys() )
model.Storages_heat = Set( within=model.Storages, initialize=list_stor_heat)
model.Storages_el = Set( within=model.Storages, initialize=list_stor_el)
model.Storages_cold = Set( within=model.Storages, initialize=list_stor_cold)

# Set for networks
list_net_el=[i for i in Networks_parameters.keys() if "El" in Networks_parameters[i]['good']]
list_net_heat=[i for i in Networks_parameters.keys() if "Heat" in Networks_parameters[i]['good']]
list_net_cold=[i for i in Networks_parameters.keys() if "Cold" in Networks_parameters[i]['good']]

model.Networks = Set (initialize=Networks_parameters.keys())
model.Networks_heat = Set( within=model.Networks, initialize=list_net_heat)
model.Networks_el = Set( within=model.Networks, initialize=list_net_el)
model.Networks_cold = Set( within=model.Networks, initialize=list_net_cold)

# Set for MES goods
model.Goods = Set (initialize=["El", "Heat", "Cold"])

# Set for machines slots
n_slots=3 # defined a priori
model.Slots = RangeSet(0, n_slots-1)

# Set of vertex for the convex hull formulation (one-degree of freedom)
model.v = Set(initialize=['Min', 'Max'])

v={}
for i in Machines_parameters.keys():
    v[i, 'Min']=Machines_parameters[i]['KIn_min']
    v[i, 'Max']=Machines_parameters[i]['KIn_max']

model.K_In=Param(model.Machines, model.v, initialize=v)

n_bpt=5
model.bins=RangeSet(0, n_bpt-1, ordered=True)
x_bpts={'HP':[0,100, 500, 2000, 10000], 'Boiler': [0,100, 1000, 10000, 50000], 'ICE': [0,1328, 13783, 26237, 38692], 'CC': [0,165, 337, 508, 680]}
cost_bpts={'HP':[0,254, 778, 2039, 6239], 'Boiler': [0,8, 69, 554, 2387], 'ICE': [0,247, 2296, 4242, 6145], 'CC': [0,248, 428, 587, 733]}
stor_x_bpts={'TES1':[0,1,2000,5000,10000], "EES1":[0, 100, 250, 500, 1000]}
stor_cost_bpts={'TES1':[0,5, 626, 1131, 1770], "EES1":[0, 50, 125, 250, 500]}

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
# Active bin variable
model.b = Var (model.Machines, model.Slots, model.bins, domain=Binary)
model.b_stor= Var(model.Storages, model.bins, domain=Binary)
# Binary variable to take into account if storage is charging (1) or discharging (0)
model.c = Var ( model.Storages, model.times, domain=Binary)
# Binary variable to take into account if electricity is sold (1) or purchased (0)
model.s = Var ( model.Networks, model.times, domain=Binary)

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

# Variable for input (fuel, el, heat, cold, etc.) [kW]
model.In = Var( model.Machines, model.Slots, model.times, domain=NonNegativeReals)

# Variable for machine output (el, heat, cold, etc.) [kW]
model.Out = Var( model.Machines, model.Slots, model.Goods, model.times, domain=NonNegativeReals)
model.Out_diss = Var( model.Machines_diss, model.Slots, model.Goods, model.times, domain=NonNegativeReals)
model.Out_us = Var( model.Machines_diss, model.Slots, model.Goods, model.times, domain=NonNegativeReals)

model.Out_Res = Var (model.Machines_Res, model.Goods, model.times, domain=NonNegativeReals)

# Energy purch (negative) /sold (positive) to the network n at time t
model.Net_exch = Var (model.Networks, model.times, domain=Reals)

# Storage variables (state of charge (SOS), charge/discharge)
model.SOS = Var (model.Storages, model.times, domain=NonNegativeReals)
model.store_net = Var(model.Storages, model.times, domain=Reals)
model.store_char = Var(model.Storages, model.times, domain=NonNegativeReals)
model.store_disch = Var(model.Storages, model.times, domain=NonNegativeReals)

# Variables to define the convex combination of the cost function linear approximation
model.gamma = Var (model.Machines, model.Slots, model.bins , domain=NonNegativeReals, bounds=(0,1))
model.gamma_stor = Var (model.Storages, model.bins , domain=NonNegativeReals, bounds=(0,1))
# Investment cost associated to machine m in slot s
model.Cinv=Var(model.Machines, model.Slots, domain=NonNegativeReals)
model.Cinv_stor=Var(model.Storages, domain=NonNegativeReals)
# Net revenues from the exchange of good with the external networks (ex. electricity --> positive if sold, negative if purchased)
model.Net_rev= Var (model.Networks, model.times, domain=Reals)


## OBJECTIVE FUNCTION
# Interest rate for the TAC calculation
CCR=0.15

def ObjFun( model ):
    return (  ( sum(model.Cinv[m, s] for m in model.Machines for s in model.Slots) + sum(
            model.Cinv_stor[es] for es in model.Storages )  )*1000 \
            + sum(model.ResArea[r]*Res_parameters[r]['InvCost'] for r in model.Machines_Res)  )*CCR + sum (
            (           - sum(model.Net_rev[n,t] for n in model.Networks) + sum(model.In[m, s , t]*Machines_parameters[m]['fuel cost']
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
model.cuts=ConstraintList()
for k,m in enumerate(Machines_parameters.keys()):
        for s in range(n_slots-1):
            if s >= k:
                model.cuts.add( model.z_design[m, s+1] <= model.z_design[m, s] )

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


#def b_bound1(model, m, s):
#    return model.b[m, s, 0] == 0
def b_bound2(model, m, s):
    return model.b[m, s, n_bpt-1]==0
#model.b_bound1_constr=Constraint(model.Machines, model.Slots, rule=b_bound1)
model.b_bound2_constr=Constraint(model.Machines, model.Slots, rule=b_bound2)

def bin_active_rule(model, m, s, b):
    if b==0:
        return model.gamma[m,s,b] <= model.b[m,s,b]
    return model.gamma[m,s,b] <= model.b[m,s,b-1] + model.b[m, s, b]
model.bin_active_constr=Constraint(model.Machines, model.Slots, model.bins, rule=bin_active_rule)
def b_zdesign_link(model, m, s):
    return sum( model.b[m,s,b] for b in model.bins) == model.z_design[m,s]
model.b_zdesing_link_constr=Constraint(model.Machines, model.Slots, rule=b_zdesign_link)
def gamma_rule(model, m, s):
    return sum( model.gamma[m, s, b] for b in model.bins) == model.z_design[m,s]
model.gamma_constr=Constraint(model.Machines, model.Slots, rule=gamma_rule)
# Convex hull formulation constraint for the size and cost of machine m in slot
def x_convex_rule(model, m, s):
    return model.x_design[m, s] == sum( model.gamma[m,s,b]*x_bpts[m][b] for b in model.bins)
model.x_convex_constr=Constraint(model.Machines, model.Slots, rule=x_convex_rule)
def cost_convex_rule(model, m, s):
    return model.Cinv[m, s] == sum( model.gamma[m,s,b]*cost_bpts[m][b] for b in model.bins)
model.Cinv_convex_constr=Constraint(model.Machines, model.Slots, rule=cost_convex_rule)


#### FOR STORAGES ANALOGOUS CONSTRAINTS ###
def b_bound2_stor(model, es):
    return model.b_stor[es, n_bpt-1]==0
model.b_bound2_stor_constr=Constraint(model.Storages, rule=b_bound2_stor)

def bin_active_stor_rule(model, es, b):
    if b==0:
        return model.gamma_stor[es,b] <= model.b_stor[es,b]
    return model.gamma_stor[es,b] <= model.b_stor[es,b-1] + model.b_stor[es, b]
model.bin_active_stor_constr=Constraint(model.Storages, model.bins, rule=bin_active_stor_rule)
def b_zdesign_stor_link(model, es):
    return sum( model.b_stor[es,b] for b in model.bins) == model.z_design_stor[es]
model.b_zdesing_stor_link_constr=Constraint(model.Storages, rule=b_zdesign_stor_link)
def gamma_stor_rule(model, es):
    return sum( model.gamma_stor[es, b] for b in model.bins) == model.z_design_stor[es]
model.gamma_stor_constr=Constraint(model.Storages, rule=gamma_stor_rule)
# Convex hull formulation constraint for the size and cost of machine m in slot
def x_convex_stor_rule(model, es):
    return model.x_design_stor[es] == sum( model.gamma_stor[es,b]*stor_x_bpts[es][b] for b in model.bins)
model.x_convex_stor_constr=Constraint(model.Storages, rule=x_convex_stor_rule)
def cost_convex_stor_rule(model, es):
    return model.Cinv_stor[es] == sum( model.gamma_stor[es,b]*stor_cost_bpts[es][b] for b in model.bins)
model.Cinv_convex_stor_constr=Constraint(model.Storages, rule=cost_convex_stor_rule)

def Stor_SOSSize_link_rule(model, es, t):
    return model.SOS[es,t] <= model.x_design_stor[es]
model.Stor_levelSize_constr=Constraint(model.Storages, model.times, rule=Stor_SOSSize_link_rule)

####

# Machine part load performance expressed as a convex combination of its operating vertexes
# Linearization of the bilinear term psi[m,s,t]=x_D[m,s]*z[m,s,t] and beta[m,s,t,v]=x_D[m,s]*alpha[m,s,t,v]
def psiBeta_rule(model, m, s, t):
    return sum( model.beta[m,s,t,v] for v in model.v-['Min']) <= model.psi[m,s,t]
def psi_zeta_rule(model, m, s, t):
    return model.psi[m,s,t] <= model.z[m,s,t] * Machines_parameters[m]['XD_max']
def psi_x_design_rule(model, m, s, t):
    return model.psi[m,s,t] <= model.x_design[m,s]
def psi_link_rule(model, m, s, t):
    return model.psi[m,s,t] >= model.x_design[m,s] - (1-model.z[m,s,t]) * Machines_parameters[m]['XD_max']
model.psi_constr1=Constraint(model.Machines, model.Slots, model.times, rule=psiBeta_rule)
model.psi_constr2=Constraint(model.Machines, model.Slots, model.times, rule=psi_zeta_rule)
model.psi_constr3=Constraint(model.Machines, model.Slots, model.times, rule=psi_x_design_rule)
model.psi_constr4=Constraint(model.Machines, model.Slots, model.times, rule=psi_link_rule)


# Definition of variables In on the basis of Beta (convex hull/combination)
def In_rule(model, m, s, t):
    return model.In[m, s, t] == model.K_In[m,'Min']*model.psi[m,s,t] + sum( model.beta[m,s,t,v]*(model.K_In[m,v]-model.K_In[m, 'Min']) for v in model.v-['Min'])
model.In_constr=Constraint(model.Machines, model.Slots, model.times, rule=In_rule)


# Min/Max energy input constraint
def machines_minIn( model, m, s, t ):
    return  model.In[m, s, t] >= model.psi[m,s,t]* Machines_parameters[m]['KIn_min']
model.machine_constr_minIn=Constraint(model.Machines, model.Slots, model.times, rule=machines_minIn)
def machines_maxIn( model, m, s, t ):
    return  model.In[m, s, t] <= model.psi[m,s,t] * Machines_parameters[m]['KIn_max']
model.machine_constr_maxIn=Constraint(model.Machines, model.Slots, model.times, rule=machines_maxIn)


# Rump Up/Max Start Up constraint
def machines_RupLim_rule( model, m, s, t):
    if t==0:
        return (model.In[m, s, t]) <= Machines_parameters[m]['RUlim']
    return (model.In[m, s, t] - model.In[m, s, t-1]) <= model.z[m,s,t-1]*Machines_parameters[m]['RUlim'] + model.z[m,s,t-1]*Machines_parameters[m]['RUSU']
model.RupLim_constr=Constraint(model.Machines, model.Slots, model.times, rule=machines_RupLim_rule)
# Rump Down/Max Shut Down constraint
def machines_RdLim_rule( model, m, s, t):
    if t==0:
        return Constraint.Skip
    return (model.In[m, s, t] - model.In[m, s, t-1]) >= -model.z[m,s,t]*Machines_parameters[m]['RDlim'] - (1-model.z[m,s,t])*Machines_parameters[m]["RDSD"]
model.RdLim_constr=Constraint(model.Machines, model.Slots, model.times, rule=machines_RdLim_rule)

# Delta on variable definition: (1) if machine m is turned on at time t, otherwise (0)
def delta_on_rule1( model, m, s, t):
    if t==0:
        return model.delta_on[m, s, t] >= (model.z[m, s, t])
    return model.delta_on[m, s, t] >= (model.z[m, s, t]-model.z[m, s, t-1])
model.delta_on_constr1=Constraint(model.Machines, model.Slots, model.times, rule=delta_on_rule1)
def delta_on_rule2( model, m, s, t):
    if t==0:
        return model.delta_on[m, s, t] <= (model.z[m, s, t] + 1)/2
    return model.delta_on[m, s, t] <= (model.z[m, s, t] - model.z[m, s, t-1] + 1)/2
model.delta_on_constr2=Constraint(model.Machines, model.Slots, model.times, rule=delta_on_rule2)

# Delta on variable definition: (1) if machine m is turned on at time t, otherwise (0)
def delta_off_rule1( model, m, s, t):
    if t==0:
        return model.delta_off[m, s, t] >= -( model.z[m, s, t] )
    return model.delta_off[m, s, t] >= -(model.z[m, s, t]-model.z[m, s, t-1])
model.delta_off_constr1=Constraint(model.Machines, model.Slots, model.times, rule=delta_off_rule1)
def delta_off_rule2( model, m, s, t):
    if t==0:
        return model.delta_off[m, s, t] <= (-model.z[m, s, t] + 1)/2
    return model.delta_off[m, s, t] <=  (-model.z[m, s, t] + model.z[m, s, t-1] + 1)/2
model.delta_off_constr2=Constraint(model.Machines, model.Slots, model.times, rule=delta_off_rule2)

# Min UP/DOWN time constraint
def min_up_rule(model, m, s, t):
    if Machines_parameters[m]['minUT']==0:
        return Constraint.Skip
    if t < Machines_parameters[m]['minUT']:
        return Constraint.Skip
    return sum(model.z[m, s, t] for t in range(t-Machines_parameters[m]['minUT'], t)
               ) >= Machines_parameters[m]['minUT']*model.delta_off[m, s, t]
model.MinUT_constr=Constraint(model.Machines, model.Slots, model.times, rule=min_up_rule)

def min_down_rule(model, m, s, t):
    if Machines_parameters[m]['minDT']==0:
        return Constraint.Skip
    if t < Machines_parameters[m]['minDT']:
        return Constraint.Skip
    return sum((1-model.z[m, s, t]) for t in range(t-Machines_parameters[m]['minDT'], t)
               ) >= Machines_parameters[m]['minDT']*model.delta_on[m, s, t]
model.MinDT_constr=Constraint(model.Machines, model.Slots, model.times, rule=min_down_rule)


# Machine ouput definition
def out_func(model, m, s, g, t):
    return model.psi[m,s,t]*(model.K_In[m,'Min']*Machines_parameters[m]['K1'][g] +Machines_parameters[m]['K2'][g]) + sum(
            Machines_parameters[m]['K1'][g]*(model.K_In[m,v]-model.K_In[m,'Min'])*model.beta[m,s,t,v] for v in model.v-['Min']) \
           + Machines_parameters[m]['K3'][g]*model.z[m,s,t]
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

# Link between storage state of charge (SOS) and charge/discharge
def storage_SOS(model, s, t):
    if t == 0:
        return model.SOS[s, t] ==  (model.store_char[s, t] - model.store_disch[s, t])*Dt
    else:
        return model.SOS[s, t] == model.SOS[s, t-1]*(1-Storage_parameters[s]['eta_sd']) + (model.store_char[s, t] - model.store_disch[s, t])*Dt
model.storage_SOS_constr = Constraint(model.Storages, model.times, rule=storage_SOS)

# Constraint to fix the storage initial level
def Stor_init_rule(model, es):
    return model.SOS[es,0] == Storage_parameters[es]["Init%"]*model.x_design_stor[es]
model.Stor_init_constr=Constraint(model.Storages, rule=Stor_init_rule)

# Storage level link
def stor_link_rule(model, s):
    return model.SOS[s,0]==model.SOS[s,T-1]
model.stor_link_constr = Constraint(model.Storages, rule=stor_link_rule)


# Define the relevant parameters to limit the network operation
for n in Networks_parameters.keys():
    Networks_parameters[n]['M1']=(max(EE_demand)+ sum(Machines_parameters[m]['XD_max']*n_slots for m in model.Machines_IntCons))
    Networks_parameters[n]['M2']=Machines_parameters['ICE']['XD_max']*n_slots

# Set of constraint to establish wether the electricity is sold or purchased
def Networks_rule1( model, n, t):
    return model.Net_exch[n,t] >= (model.s[n, t]-1) * Networks_parameters[n]["M1"]
def Networks_rule2( model, n, t):
    return model.Net_exch[n,t] <= (model.s[n, t]) * Networks_parameters[n]["M2"]
model.Networks_constr1 = Constraint(model.Networks, model.times, rule=Networks_rule1)
model.Networks_constr2 = Constraint(model.Networks, model.times, rule=Networks_rule2)

# Set of constraints to define El_tot
def Networks_rev_rule1( model, n, t):
    return model.Net_rev[n, t] <= model.Net_exch[n,t]*Networks_parameters[n]["sold_price"][t] + (1-model.s[n,t]) * Networks_parameters[n]["purch_price"][t]*Networks_parameters[n]["M1"]#[t]
def Networks_rev_rule2( model, n, t):
    return model.Net_rev[n,t] <= model.Net_exch[n,t]*Networks_parameters[n]["purch_price"][t] + (model.s[n,t]) *Networks_parameters[n]["sold_price"][t]* Networks_parameters[n]["M2"]#[t]
model.Networks_rev_constr1 = Constraint(model.Networks, model.times, rule=Networks_rev_rule1)
model.Networks_rev_constr2 = Constraint(model.Networks, model.times, rule=Networks_rev_rule2)


# subset needed for the energy balance equation only
list_Machine_IntCons_el=[i for i in Machines_parameters.keys() if Machines_parameters[i]["Internal Consumer"] if Machines_parameters[i]["In"]=="El"]
list_Machine_IntCons_heat=[i for i in Machines_parameters.keys() if Machines_parameters[i]["Internal Consumer"] if Machines_parameters[i]["In"]=="Heat"]
list_Machine_IntCons_cold=[i for i in Machines_parameters.keys() if Machines_parameters[i]["Internal Consumer"] if Machines_parameters[i]["In"]=="Cold"]

# Energy balance constraint rule
def energy_balance_rule( model, g, t ):
    if g=="El":
        return sum( model.Out[m, s, g, t] for m in model.Machines for s in model.Slots ) - \
               sum( model.Out_diss[m, s, g, t] for m in model.Machines_diss for s in model.Slots ) + \
               sum( model.Out_Res[r, g, t] for r in model.Machines_Res) - \
               sum( model.In[m, s, t] for m in list_Machine_IntCons_el for s in model.Slots) + \
               sum(model.store_net[es, t] for es in model.Storages_el) - \
               sum(model.Net_exch[n, t] for n in model.Networks_el)  == EE_demand[t]
    if g=='Heat':
        return sum( model.Out[m, s, g, t] for m in model.Machines for s in model.Slots ) - \
               sum( model.Out_diss[m, s, g, t] for m in model.Machines_diss for s in model.Slots ) + \
               sum( model.Out_Res[r, g, t] for r in model.Machines_Res) - \
               sum( model.In[m, s, t] for m in list_Machine_IntCons_heat for s in model.Slots) + \
               sum(model.store_net[es, t] for es in model.Storages_heat) - \
               sum(model.Net_exch[n, t] for n in model.Networks_heat)  == Heat_demand[t]
    if g=='Cold':
        return sum( model.Out[m, s, g, t] for m in model.Machines for s in model.Slots ) - \
               sum( model.Out_diss[m, s, g, t] for m in model.Machines_diss for s in model.Slots ) + \
               sum( model.Out_Res[r, g, t] for r in model.Machines_Res) - \
               sum( model.In[m, s, t] for m in list_Machine_IntCons_cold for s in model.Slots) + \
               sum(model.store_net[es, t] for es in model.Storages_cold) - \
               sum(model.Net_exch[n, t] for n in model.Networks_cold)  == Cold_demand[t]

# Energy Balance
model.Energy_balance_constr = Constraint(
    model.Goods,
    model.times,
    rule = energy_balance_rule
    )

## Solve PROBLEM
model.solver=SolverFactory('gurobi')
results = model.solver.solve(model, options={'mipgap':0.05},  tee=True) # tee=True to display solver output in console
results.write() # display results summary in console
#options={'mipgap':0.01},
#model.pprint()

#SolverFactory('gurobi').solve(model).write()
