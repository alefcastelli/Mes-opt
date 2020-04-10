# !!! To-Dos:
# - sono da aggiungere tutti i riferimenti temporali.
#   Per adesso si considera solo il caso in cui vi sia un tempo costante di un ora per intervallo.
# - cambiare e mettere t per gli indici temporali


# Importing pyomo module
from pyomo.environ import *
from optimization_test import *
import numpy as np
import pandas as pd
from matplotlib.pyplot import *



# Declaring parameters
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

# Available area [m2] for PV panels
Available_PV_area=10

# Electricity prices [€/kWh] # può essere usato anche un profilo <--
El_price = 0.3

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
    'Boiler1': { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat'],    'm_th': 0.976, 'q_th': -32.0,   'm_el':   0.0, 'q_el':    0.0, 'min_In':  250, 'max_In': 1000, 'RUlim': 1000, 'RDlim': 10000, 'RUSU': 1000, 'RDSD': 1000, 'minUT': 2, 'minDT': 0, 'OM':  2, 'SUcost':0.0503555, 'InvCost':173400,  'Dissipable_Heat': False, 'Internal Consumer': False },
    'Boiler2': { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat'],    'm_th': 0.976, 'q_th': -80.0,   'm_el':   0.0, 'q_el':    0.0, 'min_In':  625, 'max_In': 2500, 'RUlim': 1000, 'RDlim': 10000, 'RUSU': 2500, 'RDSD': 2500, 'minUT': 2, 'minDT': 0, 'OM':  2, 'SUcost':0.0503555, 'InvCost':173400, 'Dissipable_Heat': False, 'Internal Consumer': False },
    'Boiler3': { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat'],    'm_th': 0.976, 'q_th': -160.0,  'm_el':   0.0, 'q_el':    0.0, 'min_In': 1250, 'max_In': 5000, 'RUlim': 1000, 'RDlim': 10000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 2, 'minDT': 0, 'OM': 2, 'SUcost':0.0503555, 'InvCost':173400, 'Dissipable_Heat': False, 'Internal Consumer': False},
    'ICE1':    { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat', 'El'],    'm_th': 0.439, 'q_th': -16.82, 'm_el': 0.490, 'q_el': -171.33, 'min_In':1250, 'max_In': 2500, 'RUlim': 2500, 'RDlim': 10000, 'RUSU': 2500, 'RDSD': 2500, 'minUT': 6, 'minDT': 0, 'OM': 19, 'SUcost':0.076959, 'InvCost':1053670, 'Dissipable_Heat':  True, 'Internal Consumer': False },
    'ICE2':    { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat', 'El'],    'm_th': 0.439, 'q_th': -216.82, 'm_el': 0.490, 'q_el': -239.33, 'min_In': 3250, 'max_In': 6500, 'RUlim': 6500, 'RDlim': 10000, 'RUSU': 6500, 'RDSD': 6500, 'minUT': 6, 'minDT': 0, 'OM': 19, 'SUcost':0.076959, 'InvCost':2945670, 'Dissipable_Heat':  True, 'Internal Consumer': False },
    'ICE3':    { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat', 'El'], 'm_th': 0.439, 'q_th': -391.82, 'm_el': 0.490, 'q_el': -298.83, 'min_In': 5000, 'max_In': 10000, 'RUlim': 10000, 'RDlim': 10000, 'RUSU': 10000, 'RDSD': 10000, 'minUT': 6, 'minDT': 0, 'OM': 19, 'SUcost':0.076959, 'InvCost':4601170, 'Dissipable_Heat': True, 'Internal Consumer': False},
    'HP1':      { 'In': 'El', 'fuel cost':           0, 'goods':       ['Heat'],    'm_th': 3.59, 'q_th':    -8.0, 'm_el': 0.000, 'q_el':    0.0, 'min_In':    13, 'max_In': 100, 'RUlim': 1000, 'RDlim': 1000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 0, 'minDT': 0, 'OM':  2, 'SUcost':0.1186441, 'InvCost':452100, 'Dissipable_Heat': False, 'Internal Consumer':  True },
    'HP2':      { 'In': 'El', 'fuel cost':           0, 'goods':       ['Heat'],    'm_th': 3.59, 'q_th':    -4.0, 'm_el': 0.000, 'q_el':    0.0, 'min_In':    65, 'max_In': 500, 'RUlim': 1000, 'RDlim': 1000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 0, 'minDT': 0, 'OM':  2, 'SUcost':0.1186441, 'InvCost':595450, 'Dissipable_Heat': False, 'Internal Consumer':  True },
    'HP3':      { 'In': 'El', 'fuel cost':           0, 'goods':       ['Heat'],    'm_th': 3.59, 'q_th':    -80.0, 'm_el': 0.000, 'q_el':    0.0, 'min_In':    130, 'max_In': 1000, 'RUlim': 1000, 'RDlim': 1000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 0, 'minDT': 0, 'OM':  2, 'SUcost':0.1186441, 'InvCost':1060900, 'Dissipable_Heat': False, 'Internal Consumer':  True },
    'CC1':      {'In': 'El', 'fuel cost':            0, 'goods':       ['Cold'],    'm_th': 3.500, 'q_th':    0.00, 'm_el': 0.000, 'q_el':    0.0, 'min_In':    0, 'max_In': 1000, 'RUlim': 1000, 'RDlim': 10000, 'RUSU': 1000, 'RDSD': 1000, 'minUT': 2, 'minDT': 0, 'OM': 2, 'SUcost': 0.0, 'InvCost': 200000, 'Dissipable_Heat': False, 'Internal Consumer': True},
    'CC2':      { 'In': 'El', 'fuel cost':           0, 'goods':       ['Cold'],    'm_th': 3.500, 'q_th':    0.00, 'm_el': 0.000, 'q_el':    0.0, 'min_In':    0, 'max_In': 2500, 'RUlim': 1000, 'RDlim': 10000, 'RUSU': 2500, 'RDSD': 2500, 'minUT': 2, 'minDT': 0, 'OM':  2, 'SUcost':0.0, 'InvCost':500000,   'Dissipable_Heat': False, 'Internal Consumer':  True },
    'CC3':      { 'In': 'El', 'fuel cost':           0, 'goods':       ['Cold'],    'm_th': 3.500, 'q_th':    0.00, 'm_el': 0.000, 'q_el':    0.0, 'min_In':    0, 'max_In': 5000, 'RUlim': 1000, 'RDlim': 10000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 2, 'minDT': 0, 'OM':  2, 'SUcost':0.0, 'InvCost':1000000,   'Dissipable_Heat': False, 'Internal Consumer':  True }
    }
PV_parameters = {
    'PV': {'In': 'solar energy', 'fuel cost': 0, 'goods': ['El'],  'OM': 2, 'InvCost':300 , 'available area': 10}
}
Storage_parameters = {
     # thermal energy storage
     'TES1': { 'good': 'Heat', 'minC': 0, 'maxC': 1274, 'Init%': 0, 'eta_ch': 1, 'eta_disch': 1, 'eta_sd': 0.995, 'PmaxIn': 5000, 'PmaxOut': 5000, 'FinCval': 0.0001, 'OMxTP': 0.0001 }
     }

# Time values
# Number of time intervals
T =t

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
        #if 'El' in Machines_parameters[i]['goods']:
        #    list_Machine_el.append(i)
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

# Set for PV
model.PV_tech = Set( initialize= PV_parameters.keys())

# Set for machines slots
n_slots=3 # defined a priori
model.Slots = RangeSet(0, n_slots-1)




## VARIABLES

# Machines Variables
# Variable to define if technology t is installed in  site s
model.z_design = Var (model.Machines, model.Slots, domain=Binary)
# On/off variable
model.z = Var( model.Machines, model.Slots, model.times, domain = Binary )
# Delta on/off
model.delta_on = Var (model.Machines, model.Slots, model.times, domain=Binary)
model.delta_off = Var (model.Machines, model.Slots, model.times, domain=Binary)

# Fuel as input of the machines related to the amount at the i-th time for the j-th machine in the s-th slot
# Fuel input as power [kW]
model.fuel_In = Var( model.Machines_fuelIn, model.Slots, model.times, domain = NonNegativeReals)
# Electricity consumed by internal consumers (ex. HP, CC) of electricity
model.el_In = Var( model.Machines_elIn, model.Slots, model.times, domain=NonNegativeReals)

# Variable for Q and El produced
model.Heat_gen = Var( model.Machines_heat, model.Slots, model.times, domain=NonNegativeReals)
model.Cold_gen = Var( model.Machines_cold, model.Slots, model.times, domain=NonNegativeReals)
model.El_gen = Var( model.Machines_el, model.Slots, model.times, domain=NonNegativeReals)

# Useful variables (Heat useful = Heat gen - Heat diss)
model.Heat_useful = Var( model.Machines_heat, model.Slots, model.times, domain=NonNegativeReals)
#model.El_useful= Var( model.Machines_el, model.Slots, model.times, domain=NonNegativeReals)

# Variable for Q dissipated
model.Heat_diss = Var( model.Machines_diss, model.Slots, model.times, domain=NonNegativeReals)

# Electricity purch/sold to the network at time t
model.el_purch = Var ( model.times, domain=NonNegativeReals)
model.el_sold = Var ( model.times, domain=NonNegativeReals)
# Binary variable to take into account if electricity is sold (1) or purchased (0)
#model.s = Var ( model.times, domain=Binary) ---> aggiungere in futuro

# Storage variables (level, charge/discharge)
model.l = Var (model.Storages, model.times, domain=NonNegativeReals)
model.l[('TES1',0)].fix(0)  # storage start level = 0
model.store_charge = Var(model.Storages, model.times, domain=NonNegativeReals)
model.store_discharge = Var(model.Storages, model.times, domain=NonNegativeReals)
# Storage charge/discharge power
#model.power_in = Var (model.Storages, model.times, domain=Reals)
#model.power_out = Var (model.Storages, model.times, domain=Reals)

# Variable to define the area of PV technology installed
model.PVarea = Var (model.PV_tech, domain=NonNegativeReals)
model.el_prod_PV = Var (model.PV_tech, model.times, domain=NonNegativeReals)

# Interest rate for the NPV calculation
int_rate=0.05

## OBJECTIVE FUNCTION
'''
def ObjFun( model ):
    return sum(model.fuel_In[i, s, j]*Machines_parameters[i]['fuel cost']
               for i in model.Machines for s in model.Slots for j in model.times) + sum(
                       (model.el_purch[j]-model.el_sold[j]) for j in model.times)
'''
def ObjFun( model ):
    return - sum(model.z_design[m, s]*Machines_parameters[m]['InvCost'] for m in model.Machines for s in model.Slots) \
           - sum(model.PVarea[r]*PV_parameters[r]['InvCost'] for r in model.PV_tech) + sum(
            (          (model.el_sold[j]-model.el_purch[j])*El_price - sum(model.fuel_In[i, s , j]*Machines_parameters[i]['fuel cost']
                        for i in model.Machines_fuelIn for s in model.Slots) - sum(model.z[m, s, j]*Machines_parameters[m]['OM']
                        for m in model.Machines for s in model.Slots) - sum(model.delta_on[i, s, j]*Machines_parameters[i]['SUcost']*Machines_parameters[i]['fuel cost']
                        for i in model.Machines_fuelIn for s in model.Slots) - sum(model.delta_on[e, s, j]*Machines_parameters[e]['SUcost']*El_price
                        for e in model.Machines_elIn for s in model.Slots))/((1+int_rate)**j) for j in model.times)

model.obj = Objective(
    rule = ObjFun,
    sense = maximize
    )

## CONSTRAINTS

# Machines Constraints
# Machines range constraint
# Machines on/off binary constraint
# nota: li puoi aggregare in un unico vincolo

# The same site cannot be assigned to more than one machine: one machine per site
def sites_perMachine_rule( model, s):
    return sum(model.z_design[m,s] for m in model.Machines) <= 1
model.sites_perMachine_constr=Constraint(model.Slots, rule=sites_perMachine_rule)


# Simmetry breaking constraint on the site filling with machines
model.cuts=ConstraintList()
for k,m in enumerate(Machines_parameters.keys()):
    for s in range(n_slots-1):
        if s >= k:
            model.cuts.add( model.z_design[m, s+1] <= model.z_design[m, s] )

# Link between z design and z operational: if a machines is not installed it cannot be on in any timestep
def z_link_rule( model, m, s, t):
    return model.z[m, s, t]  <= model.z_design[m,s]
model.z_link_constr=Constraint(model.Machines, model.Slots, model.times, rule=z_link_rule)


# Min/Max energy input constraint
def machines_fuelIn_min( model, i, s, j ):
    return  model.fuel_In[i, s, j] >= Machines_parameters[i]['min_In']*model.z[i, s, j]
model.machine_constr_minFuel=Constraint(model.Machines_fuelIn, model.Slots, model.times, rule=machines_fuelIn_min)
def machines_elIn_min( model, e, s, j):
    return model.el_In[e, s, j] >= Machines_parameters[e]['min_In'] * model.z[e, s, j]
model.machine_constr_minEl=Constraint(model.Machines_elIn, model.Slots, model.times, rule=machines_elIn_min)
def machines_fuelIn_max( model, i, s, j ):
    return  model.fuel_In[i, s, j] <= Machines_parameters[i]['max_In']*model.z[i, s, j]
model.machine_constr_maxFuel=Constraint(model.Machines_fuelIn, model.Slots, model.times, rule=machines_fuelIn_max)
def machines_elIn_max( model, e, s, j):
    return model.el_In[e, s, j] <= Machines_parameters[e]['max_In'] * model.z[e, s, j]
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
            return model.el_In[h, s, j]*Machines_parameters[h]['m_th'] + model.z[h, s, j]*Machines_parameters[h]['q_th']
    if 'NG' in Machines_parameters[h]['In']:
        return  model.fuel_In[h, s, j]*Machines_parameters[h]['m_th'] + model.z[h, s, j]*Machines_parameters[h]['q_th']
def cold_production(model, c, s, j ):
    if Machines_parameters[c]['Internal Consumer']:
        if 'El' in Machines_parameters[c]['In']:
            return model.el_In[c, s, j]*Machines_parameters[c]['m_th'] + model.z[c, s, j]*Machines_parameters[c]['q_th']
# Function to obtain elecricity produced by machines
def el_production(model, p, s, j ):
    return  model.fuel_In[p, s, j]*Machines_parameters[p]['m_el'] + model.z[p, s, j]*Machines_parameters[p]['q_el']

def Qprod_rule(model, h, s, j):
    return model.Heat_gen[h, s, j] == heat_production(model, h, s, j)
model.Qprod_constr=Constraint(model.Machines_heat, model.Slots, model.times, rule=Qprod_rule)

def Coldprod_rule(model, c, s, j):
    return model.Cold_gen[c, s, j] == cold_production(model, c, s, j)
model.Coldprod_constr=Constraint(model.Machines_cold, model.Slots, model.times, rule=Coldprod_rule)

def Elprod_rule(model, p, s, j):
    return model.El_gen[p, s, j] == el_production(model, p, s, j)
model.Elprod_constr=Constraint (model.Machines_el, model.Slots, model.times, rule=Elprod_rule)

# PV panels production modelling
def ElprodPV_rule(model, r, j):
    return model.el_prod_PV[r, j] <= PV_output[j]*model.PVarea[r]
model.ElprodPV_constr=Constraint (model.PV_tech, model.times, rule=ElprodPV_rule)

def Available_area_rule(model, r):
    return model.PVarea[r] <= PV_parameters[r]["available area"]
model.Available_area_constr=Constraint (model.PV_tech, rule=Available_area_rule)

# El prod/useful constraint
#def el_us_rule( model, i, s, j ):
#    return model.Eluseful[i, s, j] == model.Elprod[i, s, j]
#model.el_us_constr = Constraint(model.Machines_el, model.Slots, model.times, rule = el_us_rule)

# Q useful constraint: useful heat if generated but not dissipated
# A constraint that the system uses to define Q diss each time step
def heat_us_rule( model, h, s, j ):
    if Machines_parameters[h]['Dissipable_Heat']:
        return model.Heat_useful[h, s, j] == model.Heat_gen[h, s, j] - model.Heat_diss[h, s, j]
    return model.Heat_useful[h, s, j] == model.Heat_gen[h, s, j]
model.Heat_us_constr = Constraint(model.Machines_heat, model.Slots, model.times, rule = heat_us_rule)


# Storage Capacity constraint
def stor_capacity_rule( model, s, j):
    return model.l[s, j] <= Storage_parameters[s]['maxC']
model.stor_capacity_constr = Constraint(model.Storages ,model.times, rule=stor_capacity_rule)
'''
# Storage Power constraint
def stor_powerIn_rule ( model, s, j):
    if j==0:
        return model.power_in[s,j] == model.l[s,j]/1
    return model.power_in[s,j] == (model.l[s,j]-model.l[s,j-1])/1  # kWh/h = kW
model.stor_powerIn_constr = Constraint(model.Storages, model.times, rule=stor_powerIn_rule)
def stor_powerOut_rule ( model, s, j):
    if j==0:
        return model.power_out[s,j] == -model.l[s,j]/1
    return model.power_out[s,j] == -(model.l[s,j]-model.l[s,j-1])/1  # kWh/h = kW
model.stor_powerOut_constr = Constraint(model.Storages, model.times, rule=stor_powerOut_rule)
'''
# Link between storage level and charge/discharge
def store_level(model, s, j):
    if j == 0:
        return (model.l[s, j] == model.store_charge[s, j]-model.store_discharge[s, j])
    else:
        return (model.l[s, j] <= model.l[s, j-1]+model.store_charge[s, j]-model.store_discharge[s, j])
model.store_level_constr = Constraint(model.Storages, model.times, rule=store_level)

# Storage level link
def stor_link_rule(model, s):
    return model.l[s,0]==model.l[s,T-1]
model.stor_link_constr = Constraint(model.Storages, rule=stor_link_rule)



# Heat balance constraint rule
def heat_balance_rule( model, j ):
    return sum( model.Heat_useful[h, s, j] for h in model.Machines_heat for s in model.Slots ) + sum(
        model.store_discharge[ss, j] for ss in model.Storages)  == Heat_demand[j] + sum(model.store_charge[ss, j] for ss in model.Storages)
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
                ) + model.el_purch[j] - model.el_sold[j] + sum(model.el_prod_PV[r,j] for r in model.PV_tech)  == EE_demand[j]
# Electricity Balance
model.El_balance_constr = Constraint(
    model.times,
    rule = el_balance_rule
    )


## Solve PROBLEM
model.solver=SolverFactory('gurobi')
results = model.solver.solve(model, options={'mipgap':0.01}, tee=True) # tee=True to display solver output in console