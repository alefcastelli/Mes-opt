# !!! To-Dos:
# - sono da aggiungere tutti i riferimenti temporali.
#   Per adesso si considera solo il caso in cui vi sia un tempo costante di un ora per intervallo.
# - cambiare e mettere t per gli indici temporali


# Importing pyomo module
from pyomo.environ import *
from optimization_test import *
import numpy as np
import pandas as pd
from matplotlib.pyplot import *



## Parameters
t=24*8 # parameter to set the dimension of the problem
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
#El_price = 0.3
El_sold_price=0.1261
El_purch_price=0.1577

# Variation penalty cost [€]
var_pen = 8

# Values of the reference PES and efficiency values
safety_ref = 0.001 # added to ensure that the constraint is satisfied
eta_ref_th = 0.9
eta_ref_el = 0.5
PES_target = 0.1
eta_target = 0.75

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
    #'Boiler1': { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat'],    'm_th': 0.976, 'q_th': -32.0,   'm_el':   0.0, 'q_el':    0.0, 'min_In':  250, 'max_In': 1000, 'RUlim': 1000, 'RDlim': 10000, 'RUSU': 1000, 'RDSD': 1000, 'minUT': 2, 'minDT': 0, 'OM':  1, 'SUcost':0.0503555, 'InvCost':173400,  'Dissipable_Heat': False, 'Internal Consumer': False, 'K1Q':0.976, 'K2Q':-0.032, 'K3Q': 4.338, 'KIn_min':0.25, 'KIn_max':1, 'XD_min':250, 'XD_max':1000 },
    #'Boiler2': { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat'],    'm_th': 0.976, 'q_th': -80.0,   'm_el':   0.0, 'q_el':    0.0, 'min_In':  625, 'max_In': 2500, 'RUlim': 1000, 'RDlim': 10000, 'RUSU': 2500, 'RDSD': 2500, 'minUT': 2, 'minDT': 0, 'OM':  2, 'SUcost':0.0503555, 'InvCost':173400, 'Dissipable_Heat': False, 'Internal Consumer': False, 'K1Q':0.976, 'K2Q':-0.032, 'K3Q': 4.338, 'KIn_min':0.25, 'KIn_max':1, 'XD_min':625, 'XD_max':2500},
    'Boiler3': { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat'],    'm_th': 0.976, 'q_th': -160.0,  'm_el':   0.0, 'q_el':    0.0, 'min_In': 1250, 'max_In': 5000, 'RUlim': 1000, 'RDlim': 10000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 2, 'minDT': 0, 'OM': 3, 'SUcost':0.0503555, 'InvCost':173400, 'Dissipable_Heat': False, 'Internal Consumer': False, 'K1Q':0.976, 'K2Q':-0.032, 'K3Q': 4.338, 'KIn_min':0.25, 'KIn_max':1, 'XD_min':0, 'XD_max':50000 },
    #'ICE1':    { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat', 'El'],    'm_th': 0.439, 'q_th': -16.82, 'm_el': 0.490, 'q_el': -171.33, 'min_In':1250, 'max_In': 2500, 'RUlim': 2500, 'RDlim': 10000, 'RUSU': 2500, 'RDSD': 2500, 'minUT': 6, 'minDT': 0, 'OM': 14, 'SUcost':0.076959, 'InvCost':1053670, 'Dissipable_Heat':  True, 'Internal Consumer': False, 'K1Q':0.439, 'K2Q':-0.005, 'K3Q':-108.18 ,'K1P':0.49, 'K2P':-0.017, 'K3P': -128.83 , 'KIn_min':0.54, 'KIn_max':1, 'XD_min':1328, 'XD_max':38692},
    #'ICE2':    { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat', 'El'],    'm_th': 0.439, 'q_th': -216.82, 'm_el': 0.490, 'q_el': -239.33, 'min_In': 3250, 'max_In': 6500, 'RUlim': 6500, 'RDlim': 10000, 'RUSU': 6500, 'RDSD': 6500, 'minUT': 6, 'minDT': 0, 'OM': 16, 'SUcost':0.076959, 'InvCost':2945670, 'Dissipable_Heat':  True, 'Internal Consumer': False, 'K1Q':0.439, 'K2Q':-0.005, 'K3Q':-108.18 ,'K1P':0.49, 'K2P':-0.017, 'K3P': -128.83, 'KIn_min':0.54, 'KIn_max':1, 'XD_min':1328, 'XD_max':38692},
    'ICE3':    { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat', 'El'], 'm_th': 0.439, 'q_th': -391.82, 'm_el': 0.490, 'q_el': -298.83, 'min_In': 5000, 'max_In': 10000, 'RUlim': 10000, 'RDlim': 10000, 'RUSU': 10000, 'RDSD': 10000, 'minUT': 6, 'minDT': 0, 'OM': 18, 'SUcost':0.076959, 'InvCost':4601170, 'Dissipable_Heat': True, 'Internal Consumer': False, 'K1Q':0.439, 'K2Q':-0.005, 'K3Q':-108.18 ,'K1P':0.49, 'K2P':-0.017, 'K3P': -128.83, 'KIn_min':0.54, 'KIn_max':1, 'XD_min':0, 'XD_max':38692},
    #'HP1':      { 'In': 'El', 'fuel cost':           0, 'goods':       ['Heat'],    'm_th': 3.59, 'q_th':    -8.0, 'm_el': 0.000, 'q_el':    0.0, 'min_In':    13, 'max_In': 100, 'RUlim': 1000, 'RDlim': 1000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 0, 'minDT': 0, 'OM':  1, 'SUcost':0.1186441, 'InvCost':452100, 'Dissipable_Heat': False, 'Internal Consumer':  True , 'K1Q':3.59, 'K2Q':-0.08, 'K3Q':0, 'KIn_min':0.13, 'KIn_max':1 , 'XD_min':100, 'XD_max':10000 },
    #'HP2':      { 'In': 'El', 'fuel cost':           0, 'goods':       ['Heat'],    'm_th': 3.59, 'q_th':    -4.0, 'm_el': 0.000, 'q_el':    0.0, 'min_In':    65, 'max_In': 500, 'RUlim': 1000, 'RDlim': 1000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 0, 'minDT': 0, 'OM':  2, 'SUcost':0.1186441, 'InvCost':595450, 'Dissipable_Heat': False, 'Internal Consumer':  True , 'K1Q':3.59, 'K2Q':-0.08, 'K3Q':0  , 'KIn_min':0.13, 'KIn_max':1, 'XD_min':100, 'XD_max':10000},
    'HP3':      { 'In': 'El', 'fuel cost':           0, 'goods':       ['Heat'],    'm_th': 3.59, 'q_th':    -80.0, 'm_el': 0.000, 'q_el':    0.0, 'min_In':    130, 'max_In': 1000, 'RUlim': 1000, 'RDlim': 1000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 0, 'minDT': 0, 'OM':  3, 'SUcost':0.1186441, 'InvCost':1060900, 'Dissipable_Heat': False, 'Internal Consumer':  True , 'K1Q':3.59, 'K2Q':-0.08, 'K3Q':0 , 'KIn_min':0.13, 'KIn_max':1, 'XD_min':0, 'XD_max':10000},
    #'CC1':      {'In': 'El', 'fuel cost':            0, 'goods':       ['Cold'],    'm_th': 3.500, 'q_th':    0.00, 'm_el': 0.000, 'q_el':    0.0, 'min_In':    0, 'max_In': 1000, 'RUlim': 1000, 'RDlim': 10000, 'RUSU': 1000, 'RDSD': 1000, 'minUT': 2, 'minDT': 0, 'OM': 1, 'SUcost': 0.0, 'InvCost': 200000, 'Dissipable_Heat': False, 'Internal Consumer': True, 'K1Q': 11.10 , 'K2Q': -0.324, 'K3Q':0, 'KIn_min':0.13, 'KIn_max':1, 'XD_min':165, 'XD_max':680},
    #'CC2':      { 'In': 'El', 'fuel cost':           0, 'goods':       ['Cold'],    'm_th': 3.500, 'q_th':    0.00, 'm_el': 0.000, 'q_el':    0.0, 'min_In':    0, 'max_In': 2500, 'RUlim': 1000, 'RDlim': 10000, 'RUSU': 2500, 'RDSD': 2500, 'minUT': 2, 'minDT': 0, 'OM':  2, 'SUcost':0.0, 'InvCost':500000,   'Dissipable_Heat': False, 'Internal Consumer':  True, 'K1Q': 11.10, 'K2Q': -0.324, 'K3Q':0 , 'KIn_min':0.13, 'KIn_max':1, 'XD_min':165, 'XD_max':680},
    'CC3':      { 'In': 'El', 'fuel cost':           0, 'goods':       ['Cold'],    'm_th': 3.500, 'q_th':    0.00, 'm_el': 0.000, 'q_el':    0.0, 'min_In':    0, 'max_In': 5000, 'RUlim': 1000, 'RDlim': 10000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 2, 'minDT': 0, 'OM':  3, 'SUcost':0.0, 'InvCost':1000000,   'Dissipable_Heat': False, 'Internal Consumer':  True, 'K1Q': 11.10, 'K2Q': -0.324, 'K3Q':0, 'KIn_min':0.13, 'KIn_max':1, 'XD_min':0, 'XD_max':680},
}
Res_parameters = {
   'PV': {'In': 'SunIrradiance',  'goods': ['El'],  'OM': 10, 'InvCost':300 , 'available area': 10} # maintenance €/m2
}
Storage_parameters = {
     # thermal energy storage
     'TES1': { 'good': 'Heat', 'minC': 0, 'maxC': 1274, 'Init%': 0, 'eta_ch': 1, 'eta_disch': 1, 'eta_sd': 1, 'PmaxIn': 5000, 'PmaxOut': 5000, 'FinCval': 0.0001, 'OMxTP': 0.0001, 'InvCost': 500, 'XD_min':0, 'XD_max':10000 },
     'EES1': {'good': 'El', 'minC': 0, 'maxC': 1000, 'Init%': 0, 'eta_ch': 1, 'eta_disch': 1, 'eta_sd': 1, 'PmaxIn': 500, 'PmaxOut': 500, 'FinCval': 0.0001, 'OMxTP': 0.0001, 'InvCost': 500, 'XD_min':0, 'XD_max':1000} # €/kWh
}

# Time values
# Number of time intervals
T =t
Dt=1 # hours each time step
######################################################################################################################################################################
# Optimization Model #################################################################################################################################################

# Create model as ConcreteModel
model = ConcreteModel()

## SETS
# Set of all time intervals - in this case must be 504
model.times = RangeSet( 0, T - 1 )

# Set of all the machines
model.Machines = Set( initialize = Machines_parameters.keys() )
list_Machine_fuelIn=[]
list_Machine_elIn=[]
#list_Machine_heatIn=[]
list_Machine_heat = []
list_Machine_cold = []
list_Machine_el = []
list_Machine_diss=[]

for i in Machines_parameters.keys():
    if 'NG' in Machines_parameters[i]['In']:
        list_Machine_fuelIn.append(i)
    if Machines_parameters[i]['Internal Consumer']:
        if "El" in Machines_parameters[i]['In']:
            list_Machine_elIn.append(i)
    if Machines_parameters[i]['Dissipable_Heat']:
        list_Machine_diss.append(i)
    if 'Heat' in Machines_parameters[i]['goods']:                               # aggiungere qua le righe per anche Cold <--
        list_Machine_heat.append(i)
    if 'Cold' in Machines_parameters[i]['goods']:
        list_Machine_cold.append(i)
    elif 'El' in Machines_parameters[i]['goods']:
        list_Machine_el.append(i)
model.Machines_fuelIn = Set( within=model.Machines, initialize= list_Machine_fuelIn)
model.Machines_elIn = Set( within=model.Machines, initialize= list_Machine_elIn)
model.Machines_heat = Set( within = model.Machines, initialize = list_Machine_heat )
model.Machines_cold = Set( within= model.Machines, initialize= list_Machine_cold)
model.Machines_el = Set( within = model.Machines, initialize = list_Machine_el )
model.Machines_diss= Set( within=model.Machines, initialize=list_Machine_diss)

# Set for storage
model.Storages = Set ( initialize = Storage_parameters.keys() )
list_TES=[]
list_EES=[]
for i in Storage_parameters.keys():
    if "Heat" in Storage_parameters[i]['good']:
        list_TES.append(i)
    if "El" in Storage_parameters[i]['good']:
        list_EES.append(i)
model.Storages_TES = Set( within=model.Storages, initialize=list_TES)
model.Storages_EES = Set( within=model.Storages, initialize=list_EES)

# Set for RES
model.Machines_Res = Set( initialize= Res_parameters.keys())

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
x_bpts={#'HP1':[100, 500, 2000, 10000], 'Boiler1': [100, 1000, 10000, 50000], 'ICE1': [1328, 13783, 26237, 38692], 'CC1': [165, 337, 508, 680],
        #'HP2':[100, 500, 2000, 10000], 'Boiler2': [100, 1000, 10000, 50000], 'ICE2': [1328, 13783, 26237, 38692], 'CC2': [165, 337, 508, 680],
        'HP3':[0,100, 500, 2000, 10000], 'Boiler3': [0,100, 1000, 10000, 50000], 'ICE3': [0,1328, 13783, 26237, 38692], 'CC3': [0,165, 337, 508, 680]
         }
cost_bpts={#'HP1':[254, 778, 2039, 6239], 'Boiler1': [8, 69, 554, 2387], 'ICE1': [247, 2296, 4242, 6145], 'CC1': [248, 428, 587, 733],
           #'HP2':[254, 778, 2039, 6239], 'Boiler2': [8, 69, 554, 2387], 'ICE2': [247, 2296, 4242, 6145], 'CC2': [248, 428, 587, 733],
           'HP3':[0,254, 778, 2039, 6239], 'Boiler3': [0,8, 69, 554, 2387], 'ICE3': [0,247, 2296, 4242, 6145], 'CC3': [0,248, 428, 587, 733]
           }
stor_x_bpts={'TES1':[0,1,2000,5000,10000]}
stor_cost_bpts={'TES1':[0,5, 626, 1131, 1770]}

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
model.b_stor= Var(model.Storages_TES, model.bins, domain=Binary)
# Binary variable to take into account if electricity is sold (1) or purchased (0)
model.s = Var ( model.times, domain=Binary)

## Continuous Variables
# Variable to define the size of unit
model.x_design= Var (model.Machines, model.Slots, domain=NonNegativeReals)
model.x_design_stor= Var(model.Storages, domain=NonNegativeReals)
# Variable to define the area of PV technology installed
model.ResArea = Var (model.Machines_Res, domain=NonNegativeReals)

# Convex hull operation control variable associated to vertex v of the performance map of machine m in site s at time t
model.beta = Var (model.Machines, model.Slots, model.times, model.v, domain=NonNegativeReals)
# Linearization variable of the product x_D[m,s] * z[m,s,t]
model.psi = Var (model.Machines, model.Slots, model.times, domain=NonNegativeReals)

# Fuel input as power [kW]
model.fuel_In = Var( model.Machines_fuelIn, model.Slots, model.times, domain = NonNegativeReals)
# Electricity consumed by internal consumers (ex. HP, CC) of electricity
model.el_In = Var( model.Machines_elIn, model.Slots, model.times, domain=NonNegativeReals)

# Variable for Q and El produced
model.Heat_gen = Var( model.Machines_heat, model.Slots, model.times, domain=NonNegativeReals)
model.Cold_gen = Var( model.Machines_cold, model.Slots, model.times, domain=NonNegativeReals)
model.El_gen = Var( model.Machines_el, model.Slots, model.times, domain=NonNegativeReals)
model.El_gen_Res = Var (model.Machines_Res, model.times, domain=NonNegativeReals)
# Useful heat (Heat useful = Heat gen - Heat diss)
model.Heat_useful = Var( model.Machines_heat, model.Slots, model.times, domain=NonNegativeReals)
# Variable for Q dissipated
model.Heat_diss = Var( model.Machines_diss, model.Slots, model.times, domain=NonNegativeReals)

# Electricity purch/sold to the network at time t
model.el_grid = Var (model.times, domain=Reals)

# Storage variables (level, charge/discharge)
model.l = Var (model.Storages, model.times, domain=NonNegativeReals)
#model.l[('TES1',0)].fix(Storage_parameters['TES1']['Init%']*Storage_parameters['TES1']['maxC'])  # storage start level = 0
#model.l[('EES1',0)].fix(Storage_parameters['EES1']['Init%']*Storage_parameters['EES1']['maxC'])  # storage start level = 0
model.store_charge = Var(model.Storages, model.times, domain=NonNegativeReals)
model.store_discharge = Var(model.Storages, model.times, domain=NonNegativeReals)

# Variables to define the convex combination of the cost function linear approximation
model.gamma = Var (model.Machines, model.Slots, model.bins , domain=NonNegativeReals, bounds=(0,1))
model.gamma_stor = Var (model.Storages_TES, model.bins , domain=NonNegativeReals, bounds=(0,1))
# Investment cost associated to machine m in slot s
model.Cinv=Var(model.Machines, model.Slots, domain=NonNegativeReals)
model.Cinv_stor=Var(model.Storages_TES, domain=NonNegativeReals)
# Revenues from the electricity (positive if sold, negative if purchased)
model.El_tot = Var (model.times, domain=Reals)


# Interest rate for the TAC calculation
CCR=0.15

## OBJECTIVE FUNCTION
def ObjFun( model ):
    return (  ( sum(model.Cinv[m, s] for m in model.Machines for s in model.Slots) + sum(
            model.Cinv_stor[es] for es in model.Storages_TES )  )*1000 \
            + sum(model.x_design_stor[es]*Storage_parameters[es]['InvCost'] for es in model.Storages_EES)
            + sum(model.ResArea[r]*Res_parameters[r]['InvCost'] for r in model.Machines_Res)  )*CCR + sum(
            (           - model.El_tot[j] + sum(model.fuel_In[i, s , j]*Machines_parameters[i]['fuel cost']
                        for i in model.Machines_fuelIn for s in model.Slots) + sum(model.z[m, s, j]*Machines_parameters[m]['OM']
                        for m in model.Machines for s in model.Slots) + sum(model.ResArea[r]*Res_parameters[r]['OM'] for r in model.Machines_Res) + sum(
                        model.delta_on[i, s, j]*Machines_parameters[i]['SUcost']*Machines_parameters[i]['fuel cost']
                        for i in model.Machines_fuelIn for s in model.Slots) + sum(model.delta_on[e, s, j]*Machines_parameters[e]['SUcost']*El_purch_price
                        for e in model.Machines_elIn for s in model.Slots)      ) for j in model.times )

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
    return model.x_design_stor[es]<= model.z_design_stor[es]*Storage_parameters[es]['XD_max']
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
model.b_bound2_stor_constr=Constraint(model.Storages_TES, rule=b_bound2_stor)

def bin_active_stor_rule(model, es, b):
    if b==0:
        return model.gamma_stor[es,b] <= model.b_stor[es,b]
    return model.gamma_stor[es,b] <= model.b_stor[es,b-1] + model.b_stor[es, b]
model.bin_active_stor_constr=Constraint(model.Storages_TES, model.bins, rule=bin_active_stor_rule)
def b_zdesign_stor_link(model, es):
    return sum( model.b_stor[es,b] for b in model.bins) == model.z_design_stor[es]
model.b_zdesing_stor_link_constr=Constraint(model.Storages_TES, rule=b_zdesign_stor_link)
def gamma_stor_rule(model, es):
    return sum( model.gamma_stor[es, b] for b in model.bins) == model.z_design_stor[es]
model.gamma_stor_constr=Constraint(model.Storages_TES, rule=gamma_stor_rule)
# Convex hull formulation constraint for the size and cost of machine m in slot
def x_convex_stor_rule(model, es):
    return model.x_design_stor[es] == sum( model.gamma_stor[es,b]*stor_x_bpts[es][b] for b in model.bins)
model.x_convex_stor_constr=Constraint(model.Storages_TES, rule=x_convex_stor_rule)
def cost_convex_stor_rule(model, es):
    return model.Cinv_stor[es] == sum( model.gamma_stor[es,b]*stor_cost_bpts[es][b] for b in model.bins)
model.Cinv_convex_stor_constr=Constraint(model.Storages_TES, rule=cost_convex_stor_rule)

def Stor_levelSize_link_rule(model, es, j):
    return model.l[es,j] <= model.x_design_stor[es]
model.Stor_levelSize_constr=Constraint(model.Storages, model.times, rule=Stor_levelSize_link_rule)
def Stor_init_rule(model, es):
    return model.l[es,0] == Storage_parameters[es]["Init%"]*model.x_design_stor[es]
model.Stor_init_constr=Constraint(model.Storages, rule=Stor_init_rule)

####

# Machine part load performance expressed as a convex combination of its operating vertexes
# Linearization of the bilinear term psi[m,s,t]=x_D[m,s]*z[m,s,t] and beta[m,s,t,v]=x_D[m,s]*alpha[m,s,t,v]
def psiBeta_rule(model, m, s, j):
    return model.psi[m,s,j]== sum( model.beta[m,s,j,v] for v in model.v)
def psi_zeta_rule(model, m, s, j):
    return model.psi[m,s,j] <= model.z[m,s,j] * Machines_parameters[m]['XD_max']
def psi_x_design_rule(model, m, s, j):
    return model.psi[m,s,j] <= model.x_design[m,s]
def psi_link_rule(model, m, s, j):
    return model.psi[m,s,j] >= model.x_design[m,s] - (1-model.z[m,s,j]) * Machines_parameters[m]['XD_max']
model.psi_constr1=Constraint(model.Machines, model.Slots, model.times, rule=psiBeta_rule)
model.psi_constr2=Constraint(model.Machines, model.Slots, model.times, rule=psi_zeta_rule)
model.psi_constr3=Constraint(model.Machines, model.Slots, model.times, rule=psi_x_design_rule)
model.psi_constr4=Constraint(model.Machines, model.Slots, model.times, rule=psi_link_rule)

# Definition of variables fuel_In and el_In on the basis of Beta (convex hull/combination)
def fuel_In_rule(model, i, s, j):
    return model.fuel_In[i, s, j] == sum( model.beta[i, s, j, v] * model.K_In[i, v] for v in model.v )
model.fuel_In_constr=Constraint(model.Machines_fuelIn, model.Slots, model.times, rule=fuel_In_rule)
def el_In_rule(model, e, s, j):
    return model.el_In[e, s, j] == sum( model.beta[e, s, j, v] * model.K_In[e, v] for v in model.v )
model.el_In_constr=Constraint(model.Machines_elIn, model.Slots, model.times, rule=el_In_rule)


# Min/Max energy input constraint
def machines_fuelIn_min( model, i, s, j ):
    return  model.fuel_In[i, s, j] >= model.psi[i,s,j]* Machines_parameters[i]['KIn_min']
model.machine_constr_minFuel=Constraint(model.Machines_fuelIn, model.Slots, model.times, rule=machines_fuelIn_min)
def machines_elIn_min( model, e, s, j):
    return model.el_In[e, s, j] >=  model.psi[e,s,j] * Machines_parameters[e]['KIn_min']
model.machine_constr_minEl=Constraint(model.Machines_elIn, model.Slots, model.times, rule=machines_elIn_min)
def machines_fuelIn_max( model, i, s, j ):
    return  model.fuel_In[i, s, j] <= model.psi[i,s,j] * Machines_parameters[i]['KIn_max']
model.machine_constr_maxFuel=Constraint(model.Machines_fuelIn, model.Slots, model.times, rule=machines_fuelIn_max)
def machines_elIn_max( model, e, s, j):
    return model.el_In[e, s, j] <= model.psi[e,s,j] * Machines_parameters[e]['KIn_max']
model.machine_constr_maxEl=Constraint(model.Machines_elIn, model.Slots, model.times, rule=machines_elIn_max)


# Rump Up constraint
# Set of all time excluding zero: model.times_not0=model.times-[0]
def machines_RupLim_rule_fuel( model, i, s, j):
    if j==0:
        return (model.fuel_In[i, s, j]) <= Machines_parameters[i]['RUlim']
    return (model.fuel_In[i, s, j] - model.fuel_In[i, s, j-1]) <= Machines_parameters[i]['RUlim']
model.RupLim_constr_fuel=Constraint(model.Machines_fuelIn, model.Slots, model.times, rule=machines_RupLim_rule_fuel)
def machines_RupLim_rule_el( model, e, s, j):
    if j==0:
        return (model.el_In[e, s, j]) <= Machines_parameters[e]['RUlim']
    return (model.el_In[e, s, j] - model.el_In[e, s, j-1]) <= Machines_parameters[e]['RUlim']
model.RupLim_constr_el=Constraint(model.Machines_elIn, model.Slots, model.times, rule=machines_RupLim_rule_el)
# Rump Down constraint
def machines_RdLim_rule_fuel( model, i, s, j):
    if j==0:
        return Constraint.Skip
    return (model.fuel_In[i, s, j-1] - model.fuel_In[i, s, j]) <= Machines_parameters[i]['RDlim']
model.RdLim_constr_fuel=Constraint(model.Machines_fuelIn, model.Slots, model.times, rule=machines_RdLim_rule_fuel)
def machines_RdLim_rule_el( model, e, s, j):
    if j==0:
        return Constraint.Skip
    return (model.el_In[e, s, j-1] - model.el_In[e, s, j]) <= Machines_parameters[e]['RUlim']
model.RdLim_constr_el=Constraint(model.Machines_elIn, model.Slots, model.times, rule=machines_RdLim_rule_el)

# Delta on variable definition: (1) if machine m is turned on at time t, otherwise (0)
def delta_on_rule1( model, m, s, j):
    if j==0:
        return model.delta_on[m, s, j] >= (model.z[m, s, j])
    return model.delta_on[m, s, j] >= (model.z[m, s, j]-model.z[m, s, j-1])
model.delta_on_constr1=Constraint(model.Machines, model.Slots, model.times, rule=delta_on_rule1)
def delta_on_rule2( model, m, s, j):
    if j==0:
        return model.delta_on[m, s, j] <= 1
    return model.delta_on[m, s, j] <= (1 - model.z[m, s, j-1])
model.delta_on_constr2=Constraint(model.Machines, model.Slots, model.times, rule=delta_on_rule2)
def delta_on_rule3( model, m, s, j):
    return model.delta_on[m, s, j] <= model.z[m, s, j]
model.delta_on_constr3=Constraint(model.Machines, model.Slots, model.times, rule=delta_on_rule3)

# Delta off variable definition: (1) if machines m is turned off at time t, otherwise (0)
def delta_off_rule1( model, m, s, j):
    if j==0:
        return model.delta_off[m, s, j] >= ( -model.z[m, s, j] )
    return model.delta_off[m, s, j] >= (model.z[m, s, j-1]-model.z[m, s, j])
model.delta_off_constr1=Constraint(model.Machines, model.Slots, model.times, rule=delta_off_rule1)
def delta_off_rule2( model, m, s, j):
    if j==0:
        return model.delta_off[m, s, j] <= 1
    return model.delta_off[m, s, j] <= 1 - model.z[m, s, j]
model.delta_off_constr2=Constraint(model.Machines, model.Slots, model.times, rule=delta_off_rule2)
def delta_off_rule3( model, m, s, j):
    if j==0:
        return model.delta_off[m, s, j] <= 0
    return model.delta_off[m, s, j] <=  model.z[m, s, j-1]
model.delta_off_constr3=Constraint(model.Machines, model.Slots, model.times, rule=delta_off_rule3)

# Min UP/DOWN time constraint
def min_up_rule(model, m, s, j):
    if Machines_parameters[m]['minUT']==0:
        return Constraint.Skip
    if j < Machines_parameters[m]['minUT']:
        return Constraint.Skip
    return sum(model.z[m, s, t] for t in range(j-Machines_parameters[m]['minUT'], j)
               ) >= Machines_parameters[m]['minUT']*model.delta_off[m, s, j]
model.MinUT_constr=Constraint(model.Machines, model.Slots, model.times, rule=min_up_rule)

def min_down_rule(model, m, s, j):
    if Machines_parameters[m]['minDT']==0:
        return Constraint.Skip
    if j < Machines_parameters[m]['minDT']:
        return Constraint.Skip
    return sum((1-model.z[m, s, t]) for t in range(j-Machines_parameters[m]['minDT'], j)
               ) >= Machines_parameters[m]['minDT']*model.delta_on[m, s, j]
model.MinDT_constr=Constraint(model.Machines, model.Slots, model.times, rule=min_down_rule)

# Function to obtain heat/cold produced by machines
def heat_production(model, h, s, j ):
    if Machines_parameters[h]['Internal Consumer']:
        if 'El' in Machines_parameters[h]['In']:
            return (  sum( model.beta[h, s, j, v]*(Machines_parameters[h]['K1Q']*model.K_In[h, v] +Machines_parameters[h]['K2Q'])
                       for v in model.v ) + model.z[h, s, j]*Machines_parameters[h]['K3Q']  )
    if 'NG' in Machines_parameters[h]['In']:
        return  sum( model.beta[h, s, j, v]*(Machines_parameters[h]['K1Q']*model.K_In[h, v] +Machines_parameters[h]['K2Q'])
                       for v in model.v ) + model.z[h, s, j]*Machines_parameters[h]['K3Q']
def cold_production(model, c, s, j ):
    if Machines_parameters[c]['Internal Consumer']:
        if 'El' in Machines_parameters[c]['In']:
            return sum( model.beta[c, s, j, v]*(Machines_parameters[c]['K1Q']*model.K_In[c, v] +Machines_parameters[c]['K2Q'])
                       for v in model.v ) + model.z[c, s, j]*Machines_parameters[c]['K3Q']
# Function to obtain elecricity produced by machines
def el_production(model, p, s, j ):
    return  sum( model.beta[p, s, j, v]*(Machines_parameters[p]['K1Q']*model.K_In[p, v] +Machines_parameters[p]['K2Q'])
                       for v in model.v ) + model.z[p, s, j]*Machines_parameters[p]['K3P']

def el_production_PV(model, r, j):
    return PV_output[j]*model.ResArea[r]

def Qprod_rule(model, h, s, j):
    return model.Heat_gen[h, s, j] == heat_production(model, h, s, j)
model.Qprod_constr=Constraint(model.Machines_heat, model.Slots, model.times, rule=Qprod_rule)

def Coldprod_rule(model, c, s, j):
    return model.Cold_gen[c, s, j] == cold_production(model, c, s, j)
model.Coldprod_constr=Constraint(model.Machines_cold, model.Slots, model.times, rule=Coldprod_rule)

def Elprod_rule(model, p, s, j):
    return model.El_gen[p, s, j] <= el_production(model, p, s, j) # se metto uguale diventa unfeasible perche??
model.Elprod_constr=Constraint (model.Machines_el, model.Slots, model.times, rule=Elprod_rule)

# PV panels electricity production modelling
def El_genRes_rule(model, r, j):
    return model.El_gen_Res[r, j] == el_production_PV(model, r, j)
model.ElgenRes_constr=Constraint(model.Machines_Res, model.times, rule=El_genRes_rule)

# Q useful constraint: useful heat if generated but not dissipated
# A constraint that the system uses to define Q diss each time step
def heat_us_rule( model, h, s, j ):
    if Machines_parameters[h]['Dissipable_Heat']:
        return model.Heat_useful[h, s, j] == model.Heat_gen[h, s, j] - model.Heat_diss[h, s, j]
    return model.Heat_useful[h, s, j] == model.Heat_gen[h, s, j]
model.Heat_us_constr = Constraint(model.Machines_heat, model.Slots, model.times, rule = heat_us_rule)

'''
# Storage Capacity constraint
def stor_capacity_rule( model, s, j):
    return model.l[s, j] <= Storage_parameters[s]['maxC']
model.stor_capacity_constr = Constraint(model.Storages ,model.times, rule=stor_capacity_rule)
'''

# Storage power limits constraint
def stor_powerIn_rule ( model, s, j):
    if j==0:
        return Constraint.Skip
    return (model.l[s,j]-model.l[s,j-1]) <= Storage_parameters[s]['PmaxIn']*Dt  # kWh= kW*h
model.stor_powerIn_constr = Constraint(model.Storages, model.times, rule=stor_powerIn_rule)
def stor_powerOut_rule ( model, s, j):
    if j==0:
        return Constraint.Skip
    return (model.l[s,j-1]-model.l[s,j]) <= Storage_parameters[s]['PmaxOut']*Dt   # kWh/h = kW
model.stor_powerOut_constr = Constraint(model.Storages, model.times, rule=stor_powerOut_rule)

# Link between storage level and charge/discharge
def store_level(model, s, j):
    if j == 0:
        return (model.l[s, j] == model.store_charge[s, j]*Storage_parameters[s]['eta_ch'] -model.store_discharge[s, j]*Storage_parameters[s]['eta_disch'])
    else:
        return model.l[s, j] == model.l[s, j-1]*(1-Storage_parameters[s]['eta_sd']) \
               + model.store_charge[s, j]*Storage_parameters[s]['eta_ch'] \
               - model.store_discharge[s, j]*Storage_parameters[s]['eta_disch']
model.store_level_constr = Constraint(model.Storages, model.times, rule=store_level)

# Storage level link
def stor_link_rule(model, s):
    return model.l[s,0]==model.l[s,T-1]
model.stor_link_constr = Constraint(model.Storages, rule=stor_link_rule)

# Set of constraint to establish wether the electricity is sold or purchased
def el_grid_rule1( model, j):
    return model.el_grid[j] >= (model.s[j]-1) * (EE_demand[j] + sum( Machines_parameters[e]['XD_max'] for e in model.Machines_elIn)*n_slots) * 1.5
def el_grid_rule2( model, j):
    return model.el_grid[j] <= (model.s[j]) * (sum( Machines_parameters[p]['XD_max'] for p in model.Machines_el)*n_slots) * 1.5
model.el_grid_constr1 = Constraint(model.times, rule=el_grid_rule1)
model.el_grid_constr2 = Constraint(model.times, rule=el_grid_rule2)

# Set of constraints to define El_tot
def El_tot_ruel1( model, j):
    return model.El_tot[j] <= model.el_grid[j]*El_sold_price + (1-model.s[j]) * (EE_demand[j] + sum(
        Machines_parameters[e]['XD_max'] for e in model.Machines_elIn)*n_slots) * 1.5 * El_purch_price
def El_tot_ruel2( model, j):
    return model.El_tot[j] <= model.el_grid[j]*El_purch_price + (model.s[j]) * (sum(
        Machines_parameters[p]['XD_max'] for p in model.Machines_el)*n_slots) * 1.5 * El_sold_price
model.El_totConstr1 = Constraint(model.times, rule=El_tot_ruel1)
model.El_totConstr2 = Constraint(model.times, rule=El_tot_ruel2)

# Heat balance constraint rule
def heat_balance_rule( model, j ):
    return sum( model.Heat_useful[h, s, j] for h in model.Machines_heat for s in model.Slots ) + sum(
        model.store_discharge[ss, j] for ss in model.Storages_TES)  == Heat_demand[j] + sum(model.store_charge[ss, j] for ss in model.Storages_TES)
# Heat Balance
model.Heat_balance_constr = Constraint(
    model.times,
    rule = heat_balance_rule
    )

# Cold balance constraint rule
def cold_balance_rule( model, j ):
    return sum( model.Cold_gen[c, s, j] for c in model.Machines_cold for s in model.Slots ) == Cold_demand[j]
# Cold Balance
model.Cold_balance_constr = Constraint(
    model.times,
    rule = cold_balance_rule
    )

# Electricity balance constrint rule
def el_balance_rule( model, j ):
    return sum( model.El_gen[p, s, j] for p in model.Machines_el for s in model.Slots
                ) - sum( model.el_In[e, s, j] for e in model.Machines_elIn for s in model.Slots
                ) -model.el_grid[j] + sum(model.El_gen_Res[r,j] for r in model.Machines_Res) + sum(
                model.store_discharge[ss, j] for ss in model.Storages_EES) - sum(
                model.store_charge[ss, j] for ss in model.Storages_EES) >= EE_demand[j]
# Electricity Balance
model.El_balance_constr = Constraint(
    model.times,
    rule = el_balance_rule
    )



## Solve PROBLEM
model.solver=SolverFactory('gurobi')
results = model.solver.solve(model, options={'mipgap':0.05},  tee=True) # tee=True to display solver output in console
#options={'mipgap':0.01},
#model.pprint()