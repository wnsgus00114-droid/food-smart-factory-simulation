(******************************************************************************
 Vendor-neutral IEC 61131-3 Structured Text reference for HTST-PLC-SHADOW-001.

 SHADOW ONLY / UNVALIDATED: this listing is maintained for cause-and-effect
 traceability and software tests. It has no hardware addresses, safety-runtime
 qualification, vendor diagnostics or deployment authorization and MUST NOT
 control physical food-processing equipment.
******************************************************************************)

TYPE E_HTST_STATE : (OFF, STARTUP, RECIRCULATE, FORWARD, CIP, TRIP); END_TYPE
TYPE E_CIP_STATE : (CIP_IDLE, PRE_RINSE, CAUSTIC, INTERMEDIATE_RINSE, ACID,
                    FINAL_RINSE, CIP_COMPLETE); END_TYPE

PROGRAM HTST_REFERENCE_SHADOW
VAR_INPUT
    ScanSeconds : LREAL;
    OperatorStart : BOOL;
    OperatorStop : BOOL;
    TripAck : BOOL;
    TripReset : BOOL;
    EmergencyStop : BOOL;
    PowerGood : BOOL;
    TemperatureSensorQualityOK : BOOL;
    CIPActive : BOOL;
    CIPReleasePermissive : BOOL;
    LowTemperature : BOOL;
    LowHoldingTime : BOOL;
    HighFlow : BOOL;
    LowDifferentialPressure : BOOL;
    SensorDisagreement : BOOL;
    FDVMismatch : BOOL;
    RegeneratorLeak : BOOL;
END_VAR
VAR_OUTPUT
    State : E_HTST_STATE := OFF;
    CIPState : E_CIP_STATE := CIP_IDLE;
    FeedPumpCommand : BOOL;
    BoosterPumpCommand : BOOL;
    CIPPumpCommand : BOOL;
    SteamEnable : BOOL;
    FDVCommandForward : BOOL;
    FDVCommandDivert : BOOL;
    CIPRinseValve : BOOL;
    CIPAlkaliValve : BOOL;
    CIPAcidValve : BOOL;
    CIPDrainValve : BOOL;
    TripLatched : BOOL;
    TripAcknowledged : BOOL;
END_VAR
VAR
    StartupElapsed : LREAL;
    ForwardConfirmElapsed : LREAL;
    CIPElapsed : LREAL;
    TripEStop : BOOL;
    TripPowerBad : BOOL;
    TripSensorBad : BOOL;
    TripFDVMismatch : BOOL;
    TripRegeneratorLeak : BOOL;
    DivertLowTemperature : BOOL;
    DivertLowHoldingTime : BOOL;
    DivertHighFlow : BOOL;
    DivertLowDP : BOOL;
    DivertSensorDisagreement : BOOL;
    DivertCIPActive : BOOL;
    DivertCIPNotReleased : BOOL;
    AnyTripActive : BOOL;
    AnyDivertActive : BOOL;
END_VAR

(* Each marker below has a one-to-one cause_effect.json/Python rule. *)
(* TRACE:TRIP_ESTOP *)
TripEStop := EmergencyStop;
(* TRACE:TRIP_POWER_BAD *)
TripPowerBad := NOT PowerGood;
(* TRACE:TRIP_SENSOR_BAD *)
TripSensorBad := NOT TemperatureSensorQualityOK;
(* TRACE:TRIP_FDV_MISMATCH *)
TripFDVMismatch := FDVMismatch;
(* TRACE:TRIP_REGENERATOR_LEAK *)
TripRegeneratorLeak := RegeneratorLeak;
(* TRACE:DIVERT_LOW_TEMPERATURE *)
DivertLowTemperature := LowTemperature;
(* TRACE:DIVERT_LOW_HOLDING_TIME *)
DivertLowHoldingTime := LowHoldingTime;
(* TRACE:DIVERT_HIGH_FLOW *)
DivertHighFlow := HighFlow;
(* TRACE:DIVERT_LOW_DIFFERENTIAL_PRESSURE *)
DivertLowDP := LowDifferentialPressure;
(* TRACE:DIVERT_SENSOR_DISAGREEMENT *)
DivertSensorDisagreement := SensorDisagreement;
(* TRACE:DIVERT_CIP_ACTIVE *)
DivertCIPActive := CIPActive;
(* TRACE:DIVERT_CIP_NOT_RELEASED *)
DivertCIPNotReleased := NOT CIPReleasePermissive;

AnyTripActive := TripEStop OR TripPowerBad OR TripSensorBad OR
                 TripFDVMismatch OR TripRegeneratorLeak;
AnyDivertActive := DivertLowTemperature OR DivertLowHoldingTime OR
                   DivertHighFlow OR DivertLowDP OR
                   DivertSensorDisagreement OR DivertCIPActive OR
                   DivertCIPNotReleased;

IF AnyTripActive THEN
    TripLatched := TRUE;
    TripAcknowledged := FALSE;
END_IF;
IF TripLatched AND TripAck THEN
    TripAcknowledged := TRUE;
END_IF;
IF TripLatched AND TripAcknowledged AND TripReset AND NOT AnyTripActive THEN
    TripLatched := FALSE;
    TripAcknowledged := FALSE;
    State := OFF;
END_IF;

IF TripLatched THEN
    State := TRIP;
    StartupElapsed := 0.0;
    ForwardConfirmElapsed := 0.0;
ELSIF OperatorStop OR NOT OperatorStart THEN
    State := OFF;
    StartupElapsed := 0.0;
    ForwardConfirmElapsed := 0.0;
    CIPElapsed := 0.0;
ELSIF CIPActive THEN
    IF State <> CIP THEN
        CIPElapsed := 0.0;
        ForwardConfirmElapsed := 0.0;
    END_IF;
    State := CIP;
    CIPElapsed := CIPElapsed + ScanSeconds;
ELSIF (State = OFF) OR (State = CIP) OR (State = TRIP) THEN
    State := STARTUP;
    StartupElapsed := ScanSeconds;
    ForwardConfirmElapsed := 0.0;
    CIPElapsed := 0.0;
ELSIF State = STARTUP THEN
    StartupElapsed := StartupElapsed + ScanSeconds;
    IF StartupElapsed >= 1.0 THEN
        State := RECIRCULATE;
        ForwardConfirmElapsed := 0.0;
    END_IF;
ELSIF State = FORWARD THEN
    IF AnyDivertActive THEN
        State := RECIRCULATE;
        ForwardConfirmElapsed := 0.0;
    END_IF;
ELSIF State = RECIRCULATE THEN
    IF AnyDivertActive THEN
        ForwardConfirmElapsed := 0.0;
    ELSE
        ForwardConfirmElapsed := ForwardConfirmElapsed + ScanSeconds;
        IF ForwardConfirmElapsed >= 1.0 THEN
            State := FORWARD;
        END_IF;
    END_IF;
END_IF;

IF State = CIP THEN
    IF CIPElapsed < 60.0 THEN
        CIPState := PRE_RINSE;
    ELSIF CIPElapsed < 240.0 THEN
        CIPState := CAUSTIC;
    ELSIF CIPElapsed < 300.0 THEN
        CIPState := INTERMEDIATE_RINSE;
    ELSIF CIPElapsed < 420.0 THEN
        CIPState := ACID;
    ELSIF CIPElapsed < 540.0 THEN
        CIPState := FINAL_RINSE;
    ELSE
        CIPState := CIP_COMPLETE;
    END_IF;
ELSE
    CIPState := CIP_IDLE;
END_IF;

(* Safe defaults are written every scan; only FORWARD energizes forward. *)
FeedPumpCommand := FALSE;
BoosterPumpCommand := FALSE;
CIPPumpCommand := FALSE;
SteamEnable := FALSE;
FDVCommandForward := FALSE;
FDVCommandDivert := TRUE;
CIPRinseValve := FALSE;
CIPAlkaliValve := FALSE;
CIPAcidValve := FALSE;
CIPDrainValve := FALSE;

CASE State OF
    STARTUP, RECIRCULATE:
        FeedPumpCommand := TRUE;
        BoosterPumpCommand := TRUE;
        SteamEnable := TRUE;
    FORWARD:
        FeedPumpCommand := TRUE;
        BoosterPumpCommand := TRUE;
        SteamEnable := TRUE;
        FDVCommandForward := TRUE;
        FDVCommandDivert := FALSE;
    CIP:
        CIPPumpCommand := CIPState <> CIP_COMPLETE;
        SteamEnable := (CIPState = CAUSTIC) OR (CIPState = ACID);
        CIPRinseValve := (CIPState = PRE_RINSE) OR
                         (CIPState = INTERMEDIATE_RINSE) OR
                         (CIPState = FINAL_RINSE);
        CIPAlkaliValve := CIPState = CAUSTIC;
        CIPAcidValve := CIPState = ACID;
        CIPDrainValve := CIPRinseValve OR (CIPState = CIP_COMPLETE);
    OFF, TRIP:
        ; (* Safe defaults remain active. *)
END_CASE;

END_PROGRAM
