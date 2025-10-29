from pydantic import BaseModel

class RouteNames(BaseModel):
    index:        str = "index"
    sim_start:    str = "sim_start"
    sim_select:   str = "sim_select"
    sim_edit_get: str = "sim_edit_get"
    sim_confirm:  str = "sim_confirm_get"

class UIStrings(BaseModel):
    title_start: str = "Simulations"
    btn_run:     str = "Run Simulation"
    btn_new:     str = "New Simulation"
    btn_edit:    str = "Edit Simulation"
    btn_sweep:   str = "Sweep Simulation"
    btn_back:    str = "Back"
    btn_next:    str = "Next"

class ContextKeys(BaseModel):
    overrides: str = "overrides"

    sim_id: str = "sim_id"

    fields: str = "fields"

    base: str = "base"

    sim: str = "sim"

    sims: str = "sims"   # controllers pass list under this key
    mode: str = "mode"   # 'run' | 'edit' | 'sweep'

class Vars(BaseModel):
    routes: RouteNames = RouteNames()
    ui:     UIStrings  = UIStrings()
    keys:   ContextKeys= ContextKeys()

VARS = Vars()
