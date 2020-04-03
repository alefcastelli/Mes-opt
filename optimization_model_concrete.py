# !!! To-Dos:
# - sono da aggiungere tutti i riferimenti temporali.
#   Per adesso si considera solo il caso in cui vi sia un tempo costante di un ora per intervallo.
# - cambiare e mettere t per gli indici temporali


# Importing pyomo module
from pyomo.environ import *
from optimization_test import *
import numpy as np
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
PV_area=10
PV_gen=PV_area*PV_output # kW

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
# machine inoput as the sum of the bought electricity and the produced one

# --> in futuro aggiungere anche il tipo di macchina e il numero di priorità
Machines_parameters = {
    #'Boiler1': { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat'],    'm_th': 0.976, 'q_th': -32.0,   'm_el':   0.0, 'q_el':    0.0, 'min_In':  250, 'max_In': 1000, 'RUlim': 1000, 'RDlim': 10000, 'RUSU': 1000, 'RDSD': 1000, 'minUT': 2, 'minDT': 0, 'OM':  2, 'Dissipable_Heat': False, 'Internal Consumer': False },
    #'Boiler2': { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat'],    'm_th': 0.976, 'q_th': -80.0,   'm_el':   0.0, 'q_el':    0.0, 'min_In':  625, 'max_In': 2500, 'RUlim': 1000, 'RDlim': 10000, 'RUSU': 2500, 'RDSD': 2500, 'minUT': 2, 'minDT': 0, 'OM':  2, 'Dissipable_Heat': False, 'Internal Consumer': False },
    #'Boiler3': { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat'],    'm_th': 0.976, 'q_th': -160.0,  'm_el':   0.0, 'q_el':    0.0, 'min_In': 1250, 'max_In': 5000, 'RUlim': 1000, 'RDlim': 10000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 2, 'minDT': 0, 'OM': 2, 'Dissipable_Heat': False, 'Internal Consumer': False},
    #'ICE1':    { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat', 'El'],    'm_th': 0.439, 'q_th': -16.82, 'm_el': 0.490, 'q_el': -171.33, 'min_In':1250, 'max_In': 2500, 'RUlim': 2500, 'RDlim': 10000, 'RUSU': 2500, 'RDSD': 2500, 'minUT': 6, 'minDT': 0, 'OM': 19, 'Dissipable_Heat':  True, 'Internal Consumer': False },
    #'ICE2':    { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat', 'El'],    'm_th': 0.439, 'q_th': -216.82, 'm_el': 0.490, 'q_el': -239.33, 'min_In': 3250, 'max_In': 6500, 'RUlim': 6500, 'RDlim': 10000, 'RUSU': 6500, 'RDSD': 6500, 'minUT': 6, 'minDT': 0, 'OM': 19, 'Dissipable_Heat':  True, 'Internal Consumer': False },
    #'ICE3':    { 'In': 'NG', 'fuel cost': Fuels['NG'], 'goods': ['Heat', 'El'], 'm_th': 0.439, 'q_th': -391.82, 'm_el': 0.490, 'q_el': -298.83, 'min_In': 5000, 'max_In': 10000, 'RUlim': 10000, 'RDlim': 10000, 'RUSU': 10000, 'RDSD': 10000, 'minUT': 6, 'minDT': 0, 'OM': 19, 'Dissipable_Heat': True, 'Internal Consumer': False},
    'HP1':      { 'In': 'El', 'fuel cost':           0, 'goods':       ['Heat'],    'm_th': 3.59, 'q_th':    -8.0, 'm_el': 0.000, 'q_el':    0.0, 'min_In':    13, 'max_In': 100, 'RUlim': 1000, 'RDlim': 1000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 0, 'minDT': 0, 'OM':  2, 'Dissipable_Heat': False, 'Internal Consumer':  True },
    'HP2':      { 'In': 'El', 'fuel cost':           0, 'goods':       ['Heat'],    'm_th': 3.59, 'q_th':    -4.0, 'm_el': 0.000, 'q_el':    0.0, 'min_In':    65, 'max_In': 500, 'RUlim': 1000, 'RDlim': 1000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 0, 'minDT': 0, 'OM':  2, 'Dissipable_Heat': False, 'Internal Consumer':  True },
    'HP3':      { 'In': 'El', 'fuel cost':           0, 'goods':       ['Heat'],    'm_th': 3.59, 'q_th':    -80.0, 'm_el': 0.000, 'q_el':    0.0, 'min_In':    130, 'max_In': 1000, 'RUlim': 1000, 'RDlim': 1000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 0, 'minDT': 0, 'OM':  2, 'Dissipable_Heat': False, 'Internal Consumer':  True }
    # 'CC':      { 'In': 'El', 'fuel cost':           0, 'goods':       ['Cold'],    'm_th': 3.500, 'q_th':    0.00, 'm_el': 0.000, 'q_el':    0.0, 'min_In':    0, 'max_In': 5000, 'RUlim': 1000, 'RDlim': 10000, 'RUSU': 5000, 'RDSD': 5000, 'minUT': 2, 'minDT': 0, 'OM':  2, 'Dissipable_Heat': False, 'Internal Consumer':  True }
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
list_Machine_int=[]
list_Machine_heat = []
list_Machine_el = []
list_Machine_diss=[]
for i in Machines_parameters.keys():
    if Machines_parameters[i]['Internal Consumer']:
        list_Machine_int.append(i)
    if Machines_parameters[i]['Dissipable_Heat']:
        list_Machine_diss.append(i)
    if 'Heat' in Machines_parameters[i]['goods']:                               # aggiungere qua le righe per anche Cold <--
        list_Machine_heat.append(i)
        if 'El' in Machines_parameters[i]['goods']:
            list_Machine_el.append(i)
    elif 'El' in Machines_parameters[i]['goods']:
        list_Machine_el.append(i)
model.Machines_heat = Set( within = model.Machines, initialize = list_Machine_heat )
model.Machines_el = Set( within = model.Machines, initialize = list_Machine_el )
model.Machines_diss= Set( within=model.Machines, initialize=list_Machine_diss)
model.Machines_int= Set( within=model.Machines, initialize=list_Machine_int)

# Set for storage
model.Storages = Set ( initialize = Storage_parameters.keys() )

# Set for machines places
n_slots=5 # defined a priori
model.Slots = RangeSet(0, n_slots-1) # let's assume only 3 sites available




## VARIABLES

# Machines Variables
# Fuel as input of the machines related to the amout at the i-th time for the j-th machine
# Fuel input as power [kW]
model.fuel_In = Var( model.Machines, model.Slots, model.times, domain = NonNegativeReals, initialize=0)

# On/off variable
model.z = Var( model.Machines, model.Slots, model.times, domain = Binary, initialize=0 )
# Delta on/off
model.delta_on = Var (model.Machines, model.Slots, model.times, domain=Binary)
model.delta_off = Var (model.Machines, model.Slots, model.times, domain=Binary)

# Variable for Q and El produced
model.Qprod = Var( model.Machines_heat, model.Slots, model.times, domain=NonNegativeReals)
model.Elprod = Var( model.Machines_el, model.Slots, model.times, domain=NonNegativeReals)

# Useful variables
model.Quseful = Var( model.Machines_heat, model.Slots, model.times, domain=NonNegativeReals)
model.Eluseful= Var( model.Machines_el, model.Slots, model.times, domain=NonNegativeReals)

# Electricity consumed by internal consumers (ex. HP)
model.Elcons = Var( model.Machines_int, model.Slots, model.times, domain=NonNegativeReals)

# Variable for Q dissipated
model.Qdiss = Var( model.Machines_diss, model.Slots, model.times, domain=NonNegativeReals)

# Electricity purch/sold to the network at time t
model.el_purch = Var ( model.times, domain=NonNegativeReals)
model.el_sold = Var ( model.times, domain=NonNegativeReals)

# Heat Storage level
model.l = Var (model.Storages, model.times, domain=NonNegativeReals)
model.l[('TES1',0)].fix(0)  # storage start level = 0
# Storage charge/discharge power
model.power_in = Var (model.Storages, model.times, domain=Reals)
model.power_out = Var (model.Storages, model.times, domain=Reals)

# Variable to define if technology t is installed in  site s
model.z_design = Var (model.Machines, model.Slots, domain=Binary)
# Variable to define the size of the technology installed
#model.x_desing = Var (model.Machines, domain= NonNegativeReals)



## OBJECTIVE FUNCTION
def ObjFun( model ):
    return sum(model.fuel_In[i, s, j]*Machines_parameters[i]['fuel cost']
               for i in model.Machines for s in model.Slots for j in model.times) + sum(
                       (model.el_purch[j]-model.el_sold[j]) for j in model.times)

model.obj = Objective(
    rule = ObjFun,
    sense = minimize
    )

## CONSTRAINTS

# Machines Constraints
# Machines range constraint
# Machines on/off binary constraint
# nota: li puoi aggregare in un unico vincolo

# The same site cannot be assigned to more than one machine
def sites_perMachine_rule( model, s):
    return sum(model.z_design[i,s] for i in model.Machines) <= 1
model.sites_perMachine_constr=Constraint(model.Slots, rule=sites_perMachine_rule)


# Simmetry breaking constraint on the site filling with machines
model.cuts=ConstraintList()
for k,i in enumerate(Machines_parameters.keys()):
    for s in range(n_slots-1):
        if s >= k:
            model.cuts.add( model.z_design[i, s+1] <= model.z_design[i, s] )

# Link between z design and z operational: if a machines is not installed it cannot be on in any timestep
def z_link_rule( model, i, s, t):
    return model.z[i, s, t]  <= model.z_design[i,s]
model.z_link_constr=Constraint(model.Machines, model.Slots, model.times, rule=z_link_rule)


# Min/Max energy input constraint
def machines_rule_min( model, i, s, j ):
    if Machines_parameters[i]['Internal Consumer']:
        return model.Elcons[i, s, j] >= Machines_parameters[i]['min_In']*model.z[i, s, j]
    return  model.fuel_In[i, s, j] >= Machines_parameters[i]['min_In']*model.z[i, s, j]
model.machine_constr_min=Constraint(model.Machines, model.Slots, model.times, rule=machines_rule_min)
def machines_rule_max( model, i, s, j ):
    if Machines_parameters[i]['Internal Consumer']:
        return model.Elcons[i, s, j] <= Machines_parameters[i]['max_In']*model.z[i, s, j]
    return  model.fuel_In[i, s, j] <= Machines_parameters[i]['max_In']*model.z[i, s, j]
model.machine_constr_max=Constraint(model.Machines, model.Slots, model.times, rule=machines_rule_max)

# Rump Up/Down constraint
# Set of all time excluding zero: model.times_not0=model.times-[0]
def machines_RupLim_rule( model, i, s, j):
    if j==0:
        return Constraint.Skip
    if Machines_parameters[i]['Internal Consumer']:
        return model.Elcons[i, s, j] - model.Elcons[i, s, j-1] <= Machines_parameters[i]['RUlim']
    return model.fuel_In[i, s, j] - model.fuel_In[i, s, j-1] <= Machines_parameters[i]['RUlim']
model.RupLim_constr=Constraint(model.Machines, model.Slots, model.times, rule=machines_RupLim_rule)

def machines_RdLim_rule( model, i, s, j):
    if j==0:
        return Constraint.Skip
    if Machines_parameters[i]['Internal Consumer']:
        return model.Elcons[i, s, j-1] - model.Elcons[i, s, j] <= Machines_parameters[i]['RDlim']
    return model.fuel_In[i, s, j-1] - model.fuel_In[i, s, j] <= Machines_parameters[i]['RDlim']
model.RdLim_constr=Constraint(model.Machines, model.Slots, model.times, rule=machines_RdLim_rule)

# Delta on variable definition
def delta_on_rule1( model, i, s, j):
    if j==0:
        return model.delta_on[i, s, j]==0
    return model.delta_on[i, s, j] >= (model.z[i, s, j]-model.z[i, s, j-1])
model.delta_on_constr1=Constraint(model.Machines, model.Slots, model.times, rule=delta_on_rule1)
def delta_on_rule2( model, i, s, j):
    if j==0:
        return Constraint.Skip
    return model.delta_on[i, s, j] <= 1 - model.z[i, s, j-1]
model.delta_on_constr2=Constraint(model.Machines, model.Slots, model.times, rule=delta_on_rule2)
def delta_on_rule3( model, i, s, j):
    return model.delta_on[i, s, j] <= model.z[i, s, j]
model.delta_on_constr3=Constraint(model.Machines, model.Slots, model.times, rule=delta_on_rule3)

# Delta off variable definition
def delta_off_rule1( model, i, s, j):
    if j==0:
        return model.delta_off[i, s, j]==0
    return model.delta_off[i, s, j] >= (model.z[i, s, j-1]-model.z[i, s, j])
model.delta_off_constr1=Constraint(model.Machines, model.Slots, model.times, rule=delta_off_rule1)
def delta_off_rule2( model, i, s, j):
    if j==0:
        return Constraint.Skip
    return model.delta_off[i, s, j] <= 1 - model.z[i, s, j]
model.delta_off_constr2=Constraint(model.Machines, model.Slots, model.times, rule=delta_off_rule2)
def delta_off_rule3( model, i, s, j):
    if j==0:
        return Constraint.Skip
    return model.delta_off[i, s, j] <=  model.z[i, s, j-1]
model.delta_off_constr3=Constraint(model.Machines, model.Slots, model.times, rule=delta_off_rule3)


# Min UP/DOWN time constraint
def min_up_rule(model, i, s, j):
    if Machines_parameters[i]['minUT']==0:
        return Constraint.Skip
    if j < Machines_parameters[i]['minUT']:
        return Constraint.Skip
    return sum(model.z[i, s, t] for t in range(j-Machines_parameters[i]['minUT'], j)
               ) >= Machines_parameters[i]['minUT']*model.delta_off[i, s, j]
model.MinUT_constr=Constraint(model.Machines, model.Slots, model.times, rule=min_up_rule)

def min_down_rule(model, i, s, j):
    if Machines_parameters[i]['minDT']==0:
        return Constraint.Skip
    if j < Machines_parameters[i]['minDT']:
        return Constraint.Skip
    return sum((1-model.z[i, s, t]) for t in range(j-Machines_parameters[i]['minDT'], j)
               ) >= Machines_parameters[i]['minDT']*model.delta_on[i, s, j]
model.MinDT_constr=Constraint(model.Machines, model.Slots, model.times, rule=min_down_rule)




# Function to obtain heat/cold produced by machines
def heat_production(model, i, s, j ):
    if Machines_parameters[i]['Internal Consumer']:
        return model.Elcons[i, s, j]*Machines_parameters[i]['m_th'] + model.z[i, s, j]*Machines_parameters[i]['q_th']
    return  model.fuel_In[i, s, j]*Machines_parameters[i]['m_th'] + model.z[i, s, j]*Machines_parameters[i]['q_th']
# Function to obtain elecricity produced by machines
def el_production(model, i, s, j ):
    return  model.fuel_In[i, s, j]*Machines_parameters[i]['m_el'] + model.z[i, s, j]*Machines_parameters[i]['q_el']

def Qprod_rule(model, i, s, j):
    return model.Qprod[i, s, j] == heat_production(model, i, s, j)
model.Qprod_constr=Constraint(model.Machines_heat, model.Slots, model.times, rule=Qprod_rule)

def Elprod_rule(model, i, s, j):
    return model.Elprod[i, s, j] == el_production(model, i, s, j)
model.Elprod_constr=Constraint (model.Machines_el, model.Slots, model.times, rule=Elprod_rule)

'''
# Q prod/useful constraint
def heat_us_rule( model, i, j ):
    return model.Quseful[i, j] <= model.Qprod[i, j]
model.Heat_us_constr = Constraint(model.Machines_heat, model.times, rule = heat_us_rule)
'''

# El prod/useful constraint
def el_us_rule( model, i, s, j ):
    return model.Eluseful[i, s, j] == model.Elprod[i, s, j]
model.el_us_constr = Constraint(model.Machines_el, model.Slots, model.times, rule = el_us_rule)

# Q useful constraint: useful heat if generated but not dissipated
# A constraint that the system uses to define Q diss each time step
def heat_diss_rule( model, i, s, j ):
    if Machines_parameters[i]['Dissipable_Heat']:
        return model.Quseful[i, s, j] == model.Qprod[i, s, j] - model.Qdiss[i, s, j]
    return model.Quseful[i, s, j] == model.Qprod[i, s, j]
model.Heat_diss_constr = Constraint(model.Machines_heat, model.Slots, model.times, rule = heat_diss_rule)



# Storage Capacity constraint
def stor_capacity_rule( model, s, j):
    return model.l[s, j] <= Storage_parameters[s]['maxC']
model.stor_capacity_constr = Constraint(model.Storages ,model.times, rule=stor_capacity_rule)

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

# Storage level link
def stor_link_rule(model, s):
    return model.l[s,0]==model.l[s,T-1]
model.stor_link_constr = Constraint(model.Storages, rule=stor_link_rule)



# Heat balance constraint rule
def heat_balance_rule( model, j, ss ):
    if j==0:
        return sum( model.Quseful[i, s, j] for i in model.Machines_heat for s in model.Slots ) == Heat_demand[j]
    return sum( model.Quseful[i, s, j] for i in model.Machines_heat for s in model.Slots ) - (model.l[ss, j]-model.l[ss, j-1]) == Heat_demand[j]
# Heat Balance
model.Heat_balance_constr = Constraint(
    model.times,
    model.Storages,
    rule = heat_balance_rule
    )

# Electricity balance constrint rule
def el_balance_rule( model, j ):
    return sum( model.Eluseful[i, s, j] for i in model.Machines_el for s in model.Slots
                ) - sum( model.Elcons[i, s, j] for i in model.Machines_int for s in model.Slots
                ) + model.el_purch[j] - model.el_sold[j] + PV_gen[j] == EE_demand[j]
# Electricity Balance
model.El_balance_constr = Constraint(
    model.times,
    rule = el_balance_rule
    )


## Solve PROBLEM
model.solver=SolverFactory('glpk')
results = model.solver.solve(model, options={'mipgap':0.05}, tee=True) # tee=True to display solver output in console