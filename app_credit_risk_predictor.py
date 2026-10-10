import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import datetime, hashlib, math, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Picsou Bank · Analyse de risque crédit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "lang" not in st.session_state:
    st.session_state.lang = "fr"
if "results" not in st.session_state:
    st.session_state.results = None

# CHARTE GRAPHIQUE
# Cobalt (structure) · Or (signature) · Menthe / Ambre / Corail (verdicts)
# Titres : Bricolage Grotesque — Texte : Figtree

C_LOW, C_MOD, C_HIGH = "#14B08F", "#F2A011", "#EE5A4B"

LEVELS = {
    "low":      dict(stroke=C_LOW,  solid="#0B7A66", tint="#DDF4EC", ink="#0B6B59"),
    "moderate": dict(stroke=C_MOD,  solid="#A85F00", tint="#FDF1D6", ink="#8A4D00"),
    "high":     dict(stroke=C_HIGH, solid="#C9301F", tint="#FDE4E0", ink="#A9261A"),
}

VERDICT_ICON = {
    "low":      '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    "moderate": '<circle cx="11" cy="11" r="6"/><path d="M20 20l-4.2-4.2"/>',
    "high":     '<path d="M6 6l12 12M18 6L6 18"/>',
}

SECTIONS = {
    "applicant":   ("#E4E8FB", "#2433A8", '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-8 8-8s8 3.6 8 8"/>'),
    "employment":  ("#FDE6E2", "#D9422F", '<rect x="3" y="7" width="18" height="13" rx="2"/><path d="M9 7V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2M3 13h18"/>'),
    "application": ("#FDF0D0", "#B97A06", '<path d="M6 3h9l4 4v14H6z"/><path d="M14 3v5h5M9 13h6M9 17h6"/>'),
    "exposure":    ("#D9F3EC", "#0E8570", '<rect x="2.5" y="5" width="19" height="14" rx="2.5"/><path d="M2.5 10h19M6 15h4"/>'),
    "history":     ("#ECE4FB", "#6B3FD1", '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>'),
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700;12..96,800&family=Figtree:wght@400;500;600;700&display=swap');

:root {
    --ink: #121A3A;
    --muted: #5B6486;
    --line: #DCE1F2;
    --paper: #F2F4FB;
    --cobalt: #2433A8;
    --cobalt-deep: #1A2580;
    --gold: #F4B83A;
}

html, body, .stApp, [data-testid="stAppViewContainer"], input, textarea, button {
    font-family: 'Figtree', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: var(--paper) !important;
    color: var(--ink);
}

header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }

.block-container {
    width: 100% !important;
    max-width: 1240px !important;
    margin: 0 auto !important;
    padding: 2.4rem 2rem 4rem !important;
}
[data-testid="stVerticalBlock"] { gap: 0.9rem !important; }

/* ── En-tête de page ── */
.pb-title {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 2.15rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.1;
    color: var(--ink);
    margin: 0;
}
.pb-sub {
    font-size: 0.95rem;
    color: var(--muted);
    margin: 0.5rem 0 0.6rem;
    max-width: 62ch;
    line-height: 1.5;
}

/* ── Cartes de formulaire ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1px solid var(--line) !important;
    border-radius: 20px !important;
    padding: 1.3rem 1.5rem 1.2rem !important;
}

.pb-sec { display: flex; align-items: center; gap: 0.85rem; margin-bottom: 0.4rem; }
.pb-sec-ic {
    width: 40px; height: 40px; border-radius: 12px; flex: none;
    display: flex; align-items: center; justify-content: center;
}
.pb-sec-t {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.12rem; font-weight: 700; letter-spacing: -0.01em; line-height: 1.2;
}
.pb-sec-d { font-size: 0.8rem; color: var(--muted); margin-top: 0.1rem; }

/* ── Champs ── */
[data-testid="stWidgetLabel"] p {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
}
[data-baseweb="input"], [data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 1.5px solid #D5DBEE !important;
    border-radius: 12px !important;
    transition: border-color .15s, box-shadow .15s;
}
[data-baseweb="input"] input { background: transparent !important; color: var(--ink) !important; font-weight: 500; }
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {
    border-color: var(--cobalt) !important;
    box-shadow: 0 0 0 3px rgba(36, 51, 168, 0.14) !important;
}
[data-baseweb="input"]:has(input:disabled) {
    background: #ECEFFC !important;
    border-color: #C9D0F3 !important;
}
[data-baseweb="input"] input:disabled {
    -webkit-text-fill-color: var(--cobalt) !important;
    color: var(--cobalt) !important;
    font-weight: 700;
}
[data-testid="InputInstructions"] { display: none !important; }
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {
    background: #F2F4FB !important; color: var(--cobalt) !important;
}

/* ── Bouton principal ── */
.stButton > button, button[data-testid="stBaseButton-primary"] {
    background: var(--cobalt) !important;
    border: 0 !important;
    border-radius: 14px !important;
    padding: 0.8rem 1.4rem !important;
    width: 100% !important;
    box-shadow: inset 0 -3px 0 var(--gold) !important;
    transition: background .15s, transform .1s;
}
.stButton > button p { color: #FFFFFF !important; font-size: 0.98rem !important; font-weight: 700 !important; }
.stButton > button:hover { background: var(--cobalt-deep) !important; }
.stButton > button:active { transform: scale(0.985); }
.stButton > button:focus-visible { outline: 3px solid var(--gold) !important; outline-offset: 2px; }

/* ── Pas de barre latérale ── */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"], [data-testid="stExpandSidebarButton"] { display: none !important; }

/* ── Panneau d'évaluation (colonne de droite, fixe au défilement) ── */
[data-testid="stColumn"]:has(.pb-anchor), [data-testid="column"]:has(.pb-anchor) {
    position: sticky;
    top: 1.2rem;
    align-self: flex-start;
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 1.4rem 1.4rem 1.5rem !important;
    max-height: calc(100vh - 2.4rem);
    overflow-y: auto;
}
@media (max-width: 900px) {
    [data-testid="stColumn"]:has(.pb-anchor), [data-testid="column"]:has(.pb-anchor) {
        position: static; max-height: none;
    }
}

.pb-ph { display: flex; align-items: center; justify-content: space-between; gap: 0.8rem; flex-wrap: wrap; }
.pb-ph-t {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.2rem; font-weight: 700; letter-spacing: -0.01em;
}
.pb-chip {
    font-size: 0.76rem; font-weight: 700; color: var(--cobalt);
    background: #E4E8FB; border-radius: 999px; padding: 0.28rem 0.7rem; white-space: nowrap;
}
.pb-chip.wait { color: var(--muted); background: #EEF0F8; font-weight: 600; }

.pb-gauge svg { display: block; width: 100%; max-width: 380px; height: auto; margin: 0.4rem auto 0; }
.pb-center { text-align: center; margin-top: 0.1rem; }
.pb-lvl {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.38rem 0.9rem; border-radius: 999px; font-weight: 700; font-size: 0.84rem;
}
.pb-lvl i { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }

.pb-verdict {
    display: flex; align-items: center; gap: 1rem;
    border-radius: 18px; padding: 1.1rem 1.2rem; color: #FFFFFF; margin-top: 1rem;
}
.pb-verdict-ic {
    width: 48px; height: 48px; border-radius: 50%; flex: none;
    background: rgba(255, 255, 255, 0.2);
    display: flex; align-items: center; justify-content: center;
}
.pb-verdict-k { font-size: 0.78rem; font-weight: 600; opacity: 0.9; }
.pb-verdict-w {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.55rem; font-weight: 700; letter-spacing: -0.01em; line-height: 1.1; margin: 0.1rem 0 0.3rem;
}
.pb-verdict-d { font-size: 0.83rem; line-height: 1.45; opacity: 0.95; }

.pb-recap {
    display: grid; grid-template-columns: repeat(3, 1fr);
    border: 1px solid var(--line); border-radius: 16px; margin-top: 1rem;
}
.pb-recap > div { padding: 0.7rem 0.85rem; min-width: 0; }
.pb-recap > div + div { border-left: 1px solid var(--line); }
.pb-recap-l { font-size: 0.72rem; font-weight: 600; color: var(--muted); }
.pb-recap-v {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1rem; font-weight: 700; margin-top: 0.15rem; white-space: nowrap;
}

.fx-h {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.05rem; font-weight: 700; letter-spacing: -0.01em; margin: 1.5rem 0 0.1rem;
}
.fx-sub { font-size: 0.8rem; font-weight: 700; margin: 0.95rem 0 0.55rem; }
.fx-row { margin-bottom: 0.7rem; }
.fx-name { display: block; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.3rem; }
.fx-track { height: 8px; background: #EDF0FA; border-radius: 99px; overflow: hidden; }
.fx-fill { height: 100%; border-radius: 99px; transform-origin: left; animation: fxgrow .7s cubic-bezier(.2,.7,.2,1) both; }
.fx-none { font-size: 0.82rem; color: var(--muted); }
@keyframes fxgrow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@media (prefers-reduced-motion: reduce) { .fx-fill { animation: none; } }

.pb-empty { text-align: center; color: var(--muted); font-size: 0.88rem; line-height: 1.5; margin: 0.3rem auto 0; max-width: 30ch; }

/* ── En-tête de marque ── */
.pb-brand { display: flex; align-items: center; gap: 0.9rem; }
.pb-brand-name {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.7rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.05; color: var(--ink);
}
.pb-brand-tag { font-size: 0.82rem; font-weight: 500; color: var(--cobalt); margin-top: 0.15rem; }
.pb-rule { height: 1px; background: var(--line); margin: 0.2rem 0 1.4rem; }

/* ── Sélecteur de langue (pastille FR / EN) ── */
[data-testid="stRadio"] { display: flex; justify-content: flex-end; }
[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 0 !important; background: #FFFFFF; border: 1.5px solid #D5DBEE;
    border-radius: 999px; padding: 3px; width: fit-content;
}
[data-testid="stRadio"] div[role="radiogroup"] label {
    margin: 0 !important; padding: 0.3rem 1rem !important;
    border-radius: 999px; cursor: pointer; transition: background .15s;
}
[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child { display: none !important; }
[data-testid="stRadio"] div[role="radiogroup"] label p {
    font-size: 0.82rem !important; font-weight: 700 !important; color: var(--muted) !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) { background: var(--cobalt); }
[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p { color: #FFFFFF !important; }
[data-testid="stRadio"] div[role="radiogroup"] label:has(input:focus-visible) { outline: 3px solid var(--gold); }

/* ── Mention légale ── */
.pb-foot {
    margin-top: 1.8rem; padding-left: 0.9rem; border-left: 3px solid var(--gold);
    font-size: 0.78rem; color: var(--muted); line-height: 1.55; max-width: 72ch;
}
</style>
""", unsafe_allow_html=True)

# TEXTES

T = {
    "en": {
        "bank_name": "Picsou Bank",
        "bank_subtitle": "Credit risk intelligence platform",
        "language": "Language",

        "page_title": "New credit application",
        "page_sub": "Fill in the applicant's file, then run the assessment. The score appears in the panel on the right.",

        "sec_applicant": "Applicant profile",
        "sec_applicant_d": "Who is applying and in what household",
        "sec_employment": "Employment and income",
        "sec_employment_d": "Stability and level of resources",
        "sec_application": "Loan requested",
        "sec_application_d": "Amount, purpose and repayment period",
        "sec_exposure": "Credit lines and exposure",
        "sec_exposure_d": "Existing commitments of the applicant",
        "sec_history": "Payment incidents",
        "sec_history_d": "Number of late payments on record",

        "f_name": "Full name",
        "f_dossier": "Dossier ID (auto-generated)",
        "f_age": "Age",
        "f_dependents": "Dependents",
        "f_marital": "Marital status",
        "f_housing": "Housing status",
        "marital_opts": ["Single", "Married", "Divorced", "Widowed", "Other"],
        "housing_opts": ["Owner", "Tenant", "Family-hosted", "Other"],

        "f_emp_status": "Employment status",
        "f_emp_seniority": "Seniority (years)",
        "f_income": "Monthly income (€)",
        "f_add_income": "Additional monthly income (€)",
        "emp_opts": ["Full-time", "Part-time", "Self-employed", "Retired", "Unemployed", "Other"],

        "f_loan_amount": "Requested amount (€)",
        "f_loan_purpose": "Loan purpose",
        "f_loan_duration": "Duration",
        "f_existing_loans": "Active credit lines and loans",
        "purpose_opts": ["Real estate", "Home renovation", "Vehicle purchase", "Education", "Debt consolidation", "Personal", "Business", "Other"],
        "duration_opts": [f"{m} months" for m in [12, 24, 36, 48, 60, 72, 84, 120, 180, 240, 300, 360]],

        "f_revolving": "Revolving credit utilization (%)",
        "f_debt_ratio": "Debt ratio (%)",
        "f_open_lines": "Open credit lines",
        "f_real_estate": "Mortgages and real estate loans",

        "f_late_30_59": "Late 30-59 days",
        "f_late_60_89": "Late 60-89 days",
        "f_late_90": "Late 90+ days",

        "tip_name": "Enter the full name to assign a unique dossier ID.",
        "tip_age": "Age of the applicant (18 to 110).",
        "tip_dependents": "Number of dependents in the household.",
        "tip_marital": "Current marital status.",
        "tip_housing": "Current housing status.",
        "tip_emp_status": "Current employment status.",
        "tip_emp_seniority": "Years in the current job.",
        "tip_income": "Main net monthly income.",
        "tip_add_income": "Secondary or property income.",
        "tip_loan_amount": "Total amount financed.",
        "tip_loan_purpose": "Intended use of the credit.",
        "tip_loan_duration": "Repayment period.",
        "tip_revolving": "Share of revolving credit lines currently used.",
        "tip_debt_ratio": "Monthly debt payments over monthly income.",
        "tip_open_lines": "Number of open credit accounts and loans.",
        "tip_real_estate": "Number of active real estate loans.",
        "tip_late_30_59": "Times the applicant was 30-59 days past due.",
        "tip_late_60_89": "Times the applicant was 60-89 days past due.",
        "tip_late_90": "Times the applicant was 90+ days past due.",

        "btn_analyze": "Run the assessment",
        "spinner": "Analyzing the file…",

        "panel_title": "Risk assessment",
        "dossier_wait": "Enter a client name…",
        "dossier_pending": "Awaiting name",
        "gauge_label": "default probability",
        "gauge_wait": "awaiting assessment",
        "empty_text": "Complete the file and run the assessment to see the score here.",

        "res_low": "Low risk",
        "res_moderate": "Moderate risk",
        "res_high": "High risk",
        "rec_label": "Proposed decision",
        "rec_approve": "Approve",
        "rec_review": "Manual review",
        "rec_decline": "Decline",
        "desc_low": "Solid profile. The file can follow the standard approval path.",
        "desc_moderate": "Mixed profile. An analyst should check the file before any decision.",
        "desc_high": "Risk above the acceptance threshold. Decline unless extra guarantees are offered.",

        "recap_amount": "Amount",
        "recap_duration": "Duration",
        "recap_income": "Income / month",
        "duration_unit": "months",

        "shap_title": "What drives the score",
        "shap_increase": "Increase the risk",
        "shap_decrease": "Reduce the risk",
        "fx_none": "No significant factor.",

        "legend_title": "How to read the score",
        "legend_low": ("Below {a} %", "Low risk, approve"),
        "legend_mod": ("{a} % to {b} %", "Moderate risk, manual review"),
        "legend_high": ("Above {b} %", "High risk, decline"),

        "final_disclaimer": "Decision-support tool built from the 'Give Me Some Credit' dataset. Results are meant to assist credit underwriters and are not an automated final decision.",

        "feat_labels": {
            "RevolvingUtilizationOfUnsecuredLines": "Revolving credit utilization",
            "age": "Borrower age",
            "TotalRetards": "Total payment delays",
            "DebtRatio": "Debt ratio",
            "MonthlyIncome_log": "Monthly income",
            "NumberOfOpenCreditLinesAndLoans": "Open credit lines",
            "NumberRealEstateLoansOrLines": "Real estate loans",
            "NumberOfDependents": "Dependents",
            "RevenuParPersonne": "Income per household member",
        },
    },

    "fr": {
        "bank_name": "Picsou Bank",
        "bank_subtitle": "Plateforme d'intelligence risque crédit",
        "language": "Langue",

        "page_title": "Nouveau dossier de crédit",
        "page_sub": "Complétez le dossier du demandeur, puis lancez l'évaluation. Le score s'affiche dans le panneau de droite.",

        "sec_applicant": "Profil du demandeur",
        "sec_applicant_d": "Qui fait la demande, et dans quel foyer",
        "sec_employment": "Situation professionnelle et revenus",
        "sec_employment_d": "Stabilité et niveau de ressources",
        "sec_application": "Prêt demandé",
        "sec_application_d": "Montant, objet et durée de remboursement",
        "sec_exposure": "Lignes de crédit et engagements",
        "sec_exposure_d": "Les engagements déjà en cours du demandeur",
        "sec_history": "Incidents de paiement",
        "sec_history_d": "Nombre de retards constatés",

        "f_name": "Nom complet",
        "f_dossier": "N° de dossier (généré automatiquement)",
        "f_age": "Âge",
        "f_dependents": "Personnes à charge",
        "f_marital": "Situation maritale",
        "f_housing": "Statut de logement",
        "marital_opts": ["Célibataire", "Marié(e)", "Divorcé(e)", "Veuf/Veuve", "Autre"],
        "housing_opts": ["Propriétaire", "Locataire", "Hébergé en famille", "Autre"],

        "f_emp_status": "Statut professionnel",
        "f_emp_seniority": "Ancienneté (années)",
        "f_income": "Revenu mensuel (€)",
        "f_add_income": "Revenus complémentaires (€)",
        "emp_opts": ["Salarié temps plein", "Salarié temps partiel", "Indépendant", "Retraité", "Sans emploi", "Autre"],

        "f_loan_amount": "Montant demandé (€)",
        "f_loan_purpose": "Objet du prêt",
        "f_loan_duration": "Durée",
        "f_existing_loans": "Prêts et crédits en cours",
        "purpose_opts": ["Immobilier", "Travaux / Rénovation", "Véhicule", "Éducation / Études", "Rachat de crédits", "Personnel", "Professionnel", "Autre"],
        "duration_opts": [f"{m} mois" for m in [12, 24, 36, 48, 60, 72, 84, 120, 180, 240, 300, 360]],

        "f_revolving": "Utilisation du crédit renouvelable (%)",
        "f_debt_ratio": "Ratio d'endettement (%)",
        "f_open_lines": "Lignes de crédit ouvertes",
        "f_real_estate": "Prêts immobiliers",

        "f_late_30_59": "Retards de 30 à 59 jours",
        "f_late_60_89": "Retards de 60 à 89 jours",
        "f_late_90": "Retards de 90 jours et +",

        "tip_name": "Saisissez le nom pour attribuer automatiquement un numéro de dossier unique.",
        "tip_age": "Âge de l'emprunteur (entre 18 et 110 ans).",
        "tip_dependents": "Nombre de personnes à charge dans le foyer.",
        "tip_marital": "Situation maritale actuelle.",
        "tip_housing": "Statut d'occupation du logement.",
        "tip_emp_status": "Statut professionnel actuel.",
        "tip_emp_seniority": "Nombre d'années dans le poste actuel.",
        "tip_income": "Revenu mensuel net principal.",
        "tip_add_income": "Revenus secondaires ou fonciers.",
        "tip_loan_amount": "Montant total financé.",
        "tip_loan_purpose": "Usage prévu pour le crédit.",
        "tip_loan_duration": "Durée d'amortissement.",
        "tip_revolving": "Part des lignes de crédit renouvelable actuellement utilisée.",
        "tip_debt_ratio": "Charges mensuelles de dette rapportées au revenu mensuel.",
        "tip_open_lines": "Comptes de crédit et prêts ouverts.",
        "tip_real_estate": "Nombre d'emprunts immobiliers actifs.",
        "tip_late_30_59": "Nombre de retards de 30 à 59 jours.",
        "tip_late_60_89": "Nombre de retards de 60 à 89 jours.",
        "tip_late_90": "Nombre de retards de 90 jours et plus.",

        "btn_analyze": "Lancer l'évaluation",
        "spinner": "Analyse du dossier en cours…",

        "panel_title": "Évaluation du risque",
        "dossier_wait": "Saisissez un nom client…",
        "dossier_pending": "Nom en attente",
        "gauge_label": "probabilité de défaut",
        "gauge_wait": "en attente d'évaluation",
        "empty_text": "Complétez le dossier puis lancez l'évaluation pour voir le score ici.",

        "res_low": "Risque faible",
        "res_moderate": "Risque modéré",
        "res_high": "Risque élevé",
        "rec_label": "Décision proposée",
        "rec_approve": "Accorder",
        "rec_review": "Étude manuelle",
        "rec_decline": "Refuser",
        "desc_low": "Profil solide. Le dossier peut suivre le circuit d'octroi standard.",
        "desc_moderate": "Profil mitigé. Un analyste doit vérifier le dossier avant toute décision.",
        "desc_high": "Risque au-dessus du seuil d'acceptation. Refus conseillé, sauf garanties complémentaires.",

        "recap_amount": "Montant",
        "recap_duration": "Durée",
        "recap_income": "Revenu / mois",
        "duration_unit": "mois",

        "shap_title": "Ce qui pèse dans le score",
        "shap_increase": "Aggravent le risque",
        "shap_decrease": "Réduisent le risque",
        "fx_none": "Aucun facteur significatif.",

        "legend_title": "Lire le score",
        "legend_low": ("Moins de {a} %", "Risque faible, accorder"),
        "legend_mod": ("De {a} % à {b} %", "Risque modéré, étude manuelle"),
        "legend_high": ("Au-delà de {b} %", "Risque élevé, refuser"),

        "final_disclaimer": "Outil d'aide à la décision développé à partir du dataset « Give Me Some Credit ». Il fournit une analyse indicative à destination des analystes crédit et ne constitue pas une décision automatique finale.",

        "feat_labels": {
            "RevolvingUtilizationOfUnsecuredLines": "Utilisation du crédit renouvelable",
            "age": "Âge de l'emprunteur",
            "TotalRetards": "Total des retards de paiement",
            "DebtRatio": "Ratio d'endettement",
            "MonthlyIncome_log": "Revenu mensuel",
            "NumberOfOpenCreditLinesAndLoans": "Lignes de crédit ouvertes",
            "NumberRealEstateLoansOrLines": "Prêts immobiliers",
            "NumberOfDependents": "Personnes à charge",
            "RevenuParPersonne": "Revenu par personne du foyer",
        },
    },
}

# CHARGEMENT DU MODÈLE

@st.cache_resource(show_spinner=False)
def load_model():
    model     = joblib.load("model/model_xgboost.pkl")
    features  = joblib.load("model/features.pkl")
    threshold = float(joblib.load("model/seuil_optimal.pkl"))
    
    # Création dynamique de l'explainer SHAP (plus besoin du fichier shap_explainer.pkl)
    explainer = shap.TreeExplainer(model)
    
    return model, features, threshold, explainer

model, FEATURES, THRESHOLD, explainer = load_model()
# OUTILS

def get_lang():
    return st.session_state.lang

def t(key):
    return T[get_lang()][key]

def html(markup):
    """Affiche un bloc HTML en une seule ligne (évite les pièges du parseur Markdown)."""
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)

def fmt_num(x, decimals=1, lang=None):
    s = f"{x:.{decimals}f}"
    return s.replace(".", ",") if (lang or get_lang()) == "fr" else s

def money(v, lang):
    s = f"{v:,.0f}"
    if lang == "fr":
        return s.replace(",", "\u202f") + "\u00a0€"
    return "€" + s

def generate_dossier_id(name):
    """Identifiant de dossier unique, généré dès la saisie du nom."""
    if not name or not name.strip():
        return None
    cleaned = name.strip().upper()
    initials = "".join(part[0] for part in cleaned.split() if part)[:3] or "PB"
    num_hash = int(hashlib.md5(cleaned.encode()).hexdigest(), 16) % 900000 + 100000
    return f"PB-{datetime.datetime.now().year}-{initials}-{num_hash}"

def risk_level(prob):
    if prob < 0.35:
        return "low"
    elif prob < THRESHOLD:
        return "moderate"
    return "high"

REC_KEY = {"low": "rec_approve", "moderate": "rec_review", "high": "rec_decline"}

def prepare_features(form):
    total_retards       = form["late_30_59"] + form["late_60_89"] + form["late_90"]
    monthly_income_log  = np.log1p(form["income"])
    revenu_par_personne = form["income"] / (form["dependents"] + 1)

    data = {
        "RevolvingUtilizationOfUnsecuredLines": form["revolving"] / 100.0,
        "age": form["age"],
        "TotalRetards": total_retards,
        "DebtRatio": form["debt_ratio"] / 100.0,
        "MonthlyIncome_log": monthly_income_log,
        "NumberOfOpenCreditLinesAndLoans": form["open_lines"],
        "NumberRealEstateLoansOrLines": form["real_estate"],
        "NumberOfDependents": form["dependents"],
        "RevenuParPersonne": revenu_par_personne,
    }
    return pd.DataFrame([data])[FEATURES]

def get_shap_factors(shap_vals, features_list, lang):
    labels = T[lang]["feat_labels"]
    factors = []
    for i, feat in enumerate(features_list):
        v = shap_vals[i]
        factors.append({
            "feature": feat,
            "label": labels.get(feat, feat),
            "value": v,
            "abs_value": abs(v),
            "direction": "up" if v > 0 else "down",
        })
    factors.sort(key=lambda x: x["abs_value"], reverse=True)
    increasing = [f for f in factors if f["direction"] == "up"]
    decreasing = [f for f in factors if f["direction"] == "down"]
    return increasing, decreasing

# JAUGE SVG (nette à toutes les tailles, sans matplotlib)

def _pt(frac, r, cx=150, cy=150):
    a = math.pi * (1 - frac)
    return cx + r * math.cos(a), cy - r * math.sin(a)

def _arc(f0, f1, r, color, width, opacity=1.0):
    x0, y0 = _pt(f0, r)
    x1, y1 = _pt(f1, r)
    return (f'<path d="M{x0:.2f} {y0:.2f} A{r} {r} 0 0 1 {x1:.2f} {y1:.2f}" fill="none" '
            f'stroke="{color}" stroke-width="{width}" opacity="{opacity}"/>')

def gauge_svg(prob, lang):
    cx, cy, r, w = 150, 150, 108, 20
    z1 = 0.35
    z2 = max(THRESHOLD, z1)
    zones = [(0.0, z1, C_LOW), (z1, z2, C_MOD), (z2, 1.0, C_HIGH)]
    gap = 0.008
    parts = []

    for a, b, col in zones:
        if b - a < 0.004:
            continue
        a2 = a + (gap / 2 if a > 0 else 0)
        b2 = b - (gap / 2 if b < 1 else 0)
        parts.append(_arc(a2, b2, r, col, w, 0.22))
        if prob is not None and prob > a2:
            parts.append(_arc(a2, min(b2, prob), r, col, w, 1.0))

    ticks = [z1] + ([z2] if z2 - z1 > 0.04 else [])
    for tk in ticks:
        x, y = _pt(tk, r + w / 2 + 14)
        parts.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" dominant-baseline="middle" '
                     f'font-size="10.5" font-weight="600" fill="#6B7396">{tk*100:.0f}\u00a0%</text>')

    if prob is not None:
        color = LEVELS[risk_level(prob)]["stroke"]
        kx, ky = _pt(prob, r)
        parts.append(f'<circle cx="{kx:.2f}" cy="{ky:.2f}" r="12" fill="#FFFFFF" stroke="{color}" stroke-width="5"/>')
        big = fmt_num(prob * 100, 1, lang) + "\u00a0%"
        small = T[lang]["gauge_label"]
    else:
        big = "–"
        small = T[lang]["gauge_wait"]

    parts.append(f'<text x="{cx}" y="126" text-anchor="middle" font-family="Bricolage Grotesque, sans-serif" '
                 f'font-size="40" font-weight="700" fill="#121A3A">{big}</text>')
    parts.append(f'<text x="{cx}" y="148" text-anchor="middle" font-size="12" font-weight="500" fill="#5B6486">{small}</text>')
    parts.append(f'<text x="{cx - r}" y="171" text-anchor="middle" font-size="10.5" font-weight="600" fill="#6B7396">0\u00a0%</text>')
    parts.append(f'<text x="{cx + r}" y="171" text-anchor="middle" font-size="10.5" font-weight="600" fill="#6B7396">100\u00a0%</text>')

    label = f"{T[lang]['gauge_label']}" + (f" {big}" if prob is not None else "")
    return (f'<div class="pb-gauge"><svg viewBox="0 0 300 178" role="img" aria-label="{label}" '
            f'xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg></div>')

# ================================================================
# BLOCS D'AFFICHAGE
# ================================================================

def section_head(key, title, desc):
    bg, fg, icon = SECTIONS[key]
    html(
        f'<div class="pb-sec"><div class="pb-sec-ic" style="background:{bg};color:{fg}">'
        f'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{icon}</svg></div>'
        f'<div><div class="pb-sec-t">{title}</div><div class="pb-sec-d">{desc}</div></div></div>'
    )

def factors_block(items, color, max_abs, title, none_txt):
    if not items:
        rows = f'<div class="fx-none">{none_txt}</div>'
    else:
        rows = ""
        for f in items[:3]:
            pct = max(6, int(min(f["abs_value"] / max_abs * 100, 100)))
            rows += (f'<div class="fx-row"><span class="fx-name">{f["label"]}</span>'
                     f'<div class="fx-track"><div class="fx-fill" style="width:{pct}%;background:{color}"></div></div></div>')
    return f'<div class="fx-sub" style="color:{color}">{title}</div>{rows}'

def empty_html(lang):
    return f'<div>{gauge_svg(None, lang)}<div class="pb-empty">{T[lang]["empty_text"]}</div></div>'

def result_html(res, lang):
    prob = res["prob"]
    key = risk_level(prob)
    L = LEVELS[key]
    rec = T[lang][REC_KEY[key]]
    desc = T[lang]["desc_" + {"low": "low", "moderate": "moderate", "high": "high"}[key]]
    lvl_txt = T[lang]["res_" + key]
    icon = VERDICT_ICON[key]
    recap = res["recap"]

    inc, dec = get_shap_factors(res["shap_vals"], FEATURES, lang)
    max_abs = max([f["abs_value"] for f in inc + dec], default=1) or 1

    return (
        '<div>'
        + gauge_svg(prob, lang)
        + f'<div class="pb-center"><span class="pb-lvl" style="background:{L["tint"]};color:{L["ink"]}">'
          f'<i style="background:{L["stroke"]}"></i>{lvl_txt}</span></div>'
        + f'<div class="pb-verdict" style="background:{L["solid"]}">'
          f'<div class="pb-verdict-ic"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" '
          f'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">{icon}</svg></div>'
          f'<div><div class="pb-verdict-k">{T[lang]["rec_label"]}</div>'
          f'<div class="pb-verdict-w">{rec}</div><div class="pb-verdict-d">{desc}</div></div></div>'
        + '<div class="pb-recap">'
          f'<div><div class="pb-recap-l">{T[lang]["recap_amount"]}</div><div class="pb-recap-v">{money(recap["amount"], lang)}</div></div>'
          f'<div><div class="pb-recap-l">{T[lang]["recap_duration"]}</div><div class="pb-recap-v">{recap["months"]}\u00a0{T[lang]["duration_unit"]}</div></div>'
          f'<div><div class="pb-recap-l">{T[lang]["recap_income"]}</div><div class="pb-recap-v">{money(recap["income"], lang)}</div></div>'
          '</div>'
        + f'<div class="fx-h">{T[lang]["shap_title"]}</div>'
        + factors_block(inc, C_HIGH, max_abs, T[lang]["shap_increase"], T[lang]["fx_none"])
        + factors_block(dec, C_LOW, max_abs, T[lang]["shap_decrease"], T[lang]["fx_none"])
        + '</div>'
    )

# LANGUE ET LOGO

def set_lang():
    """Callback du sélecteur : exécuté avant le rerun, la langue est donc à jour dès le haut de page."""
    st.session_state.lang = "fr" if st.session_state.lang_switch == "FR" else "en"

lang = st.session_state.lang

LOGO_SVG = (
    '<svg width="{s}" height="{s}" viewBox="0 0 46 46" aria-hidden="true">'
    '<circle cx="23" cy="23" r="22" fill="#F4B83A"/>'
    '<path d="M23 9.5 37 17.2H9z" fill="#1A2580"/>'
    '<rect x="10" y="18.6" width="26" height="2.2" rx="0.6" fill="#1A2580"/>'
    '<rect x="12.4" y="22.4" width="3.6" height="9.6" rx="0.6" fill="#1A2580"/>'
    '<rect x="18.4" y="22.4" width="3.6" height="9.6" rx="0.6" fill="#1A2580"/>'
    '<rect x="24"   y="22.4" width="3.6" height="9.6" rx="0.6" fill="#1A2580"/>'
    '<rect x="30"   y="22.4" width="3.6" height="9.6" rx="0.6" fill="#1A2580"/>'
    '<rect x="9"    y="33.4" width="28" height="3" rx="0.8" fill="#1A2580"/>'
    '</svg>'
)

# APPLICATION

def main_app():
    lang = get_lang()

    #En-tête : logo + nom à gauche, sélecteur de langue à droite ──
    try:
        h_brand, h_lang = st.columns([4, 1], vertical_alignment="center")
    except TypeError:  # anciennes versions de Streamlit
        h_brand, h_lang = st.columns([4, 1])
    with h_brand:
        html(
            '<div class="pb-brand">'
            + LOGO_SVG.format(s=54) +
            f'<div><div class="pb-brand-name">{T[lang]["bank_name"]}</div>'
            f'<div class="pb-brand-tag">{T[lang]["bank_subtitle"]}</div></div></div>'
        )
    with h_lang:
        st.radio(T[lang]["language"], ["FR", "EN"],
                 index=0 if lang == "fr" else 1,
                 horizontal=True, key="lang_switch",
                 label_visibility="collapsed", on_change=set_lang)
    html('<div class="pb-rule"></div>')

    html(f'<div><div class="pb-title">{T[lang]["page_title"]}</div>'
         f'<div class="pb-sub">{T[lang]["page_sub"]}</div></div>')

    left, right = st.columns([1.55, 1], gap="large")

    #Colonne gauche : formulaire
    with left:
        with st.container(border=True):
            section_head("applicant", T[lang]["sec_applicant"], T[lang]["sec_applicant_d"])
            c1, c2 = st.columns(2)
            with c1:
                full_name = st.text_input(T[lang]["f_name"], placeholder="ex. Jean Dupont", help=T[lang]["tip_name"])
            dossier_id = generate_dossier_id(full_name)
            with c2:
                st.text_input(T[lang]["f_dossier"], value=dossier_id or T[lang]["dossier_wait"], disabled=True)

            c3, c4, c5 = st.columns(3)
            with c3:
                age = st.number_input(T[lang]["f_age"], min_value=18, max_value=110, value=38, help=T[lang]["tip_age"])
            with c4:
                dependents = st.number_input(T[lang]["f_dependents"], min_value=0, max_value=20, value=1, help=T[lang]["tip_dependents"])
            with c5:
                marital = st.selectbox(T[lang]["f_marital"], options=T[lang]["marital_opts"], index=1, help=T[lang]["tip_marital"])
            housing = st.selectbox(T[lang]["f_housing"], options=T[lang]["housing_opts"], index=0, help=T[lang]["tip_housing"])

        with st.container(border=True):
            section_head("employment", T[lang]["sec_employment"], T[lang]["sec_employment_d"])
            c6, c7 = st.columns(2)
            with c6:
                emp_status = st.selectbox(T[lang]["f_emp_status"], options=T[lang]["emp_opts"], index=0, help=T[lang]["tip_emp_status"])
            with c7:
                emp_seniority = st.number_input(T[lang]["f_emp_seniority"], min_value=0, max_value=60, value=5, help=T[lang]["tip_emp_seniority"])
            c8, c9 = st.columns(2)
            with c8:
                income = st.number_input(T[lang]["f_income"], min_value=0, max_value=500000, value=3200, step=100, help=T[lang]["tip_income"])
            with c9:
                add_income = st.number_input(T[lang]["f_add_income"], min_value=0, max_value=500000, value=0, step=100, help=T[lang]["tip_add_income"])

        with st.container(border=True):
            section_head("application", T[lang]["sec_application"], T[lang]["sec_application_d"])
            c10, c11 = st.columns(2)
            with c10:
                loan_amount = st.number_input(T[lang]["f_loan_amount"], min_value=0, max_value=10000000, value=15000, step=500, help=T[lang]["tip_loan_amount"])
            with c11:
                loan_purpose = st.selectbox(T[lang]["f_loan_purpose"], options=T[lang]["purpose_opts"], index=0, help=T[lang]["tip_loan_purpose"])
            c12, c13 = st.columns(2)
            with c12:
                loan_duration = st.selectbox(T[lang]["f_loan_duration"], options=T[lang]["duration_opts"], index=4, help=T[lang]["tip_loan_duration"])
            with c13:
                open_credit = st.number_input(T[lang]["f_existing_loans"], min_value=0, max_value=58, value=3, help=T[lang]["tip_open_lines"])

        with st.container(border=True):
            section_head("exposure", T[lang]["sec_exposure"], T[lang]["sec_exposure_d"])
            c14, c15 = st.columns(2)
            with c14:
                revolving = st.number_input(T[lang]["f_revolving"], min_value=0.0, max_value=100.0, value=22.0, help=T[lang]["tip_revolving"])
            with c15:
                debt_ratio = st.number_input(T[lang]["f_debt_ratio"], min_value=0.0, max_value=100.0, value=28.0, help=T[lang]["tip_debt_ratio"])
            real_estate = st.number_input(T[lang]["f_real_estate"], min_value=0, max_value=54, value=1, help=T[lang]["tip_real_estate"])

        with st.container(border=True):
            section_head("history", T[lang]["sec_history"], T[lang]["sec_history_d"])
            c17, c18, c19 = st.columns(3)
            with c17:
                late_30_59 = st.number_input(T[lang]["f_late_30_59"], min_value=0, max_value=13, value=0, help=T[lang]["tip_late_30_59"])
            with c18:
                late_60_89 = st.number_input(T[lang]["f_late_60_89"], min_value=0, max_value=11, value=0, help=T[lang]["tip_late_60_89"])
            with c19:
                late_90 = st.number_input(T[lang]["f_late_90"], min_value=0, max_value=17, value=0, help=T[lang]["tip_late_90"])

    #Colonne droite : panneau d'évaluation 
    with right:
        head_slot = st.empty()
        submit = st.button(T[lang]["btn_analyze"], type="primary")
        body_slot = st.empty()

        if submit:
            form_data = {
                "age": int(age),
                "dependents": int(dependents),
                "income": float(income),
                "revolving": float(revolving),
                "debt_ratio": float(debt_ratio),
                "open_lines": int(open_credit),
                "real_estate": int(real_estate),
                "late_30_59": int(late_30_59),
                "late_60_89": int(late_60_89),
                "late_90": int(late_90),
            }

            with st.spinner(T[lang]["spinner"]):
                X_client  = prepare_features(form_data)
                prob      = float(model.predict_proba(X_client)[0][1])
                sv        = explainer.shap_values(X_client)
                shap_vals = sv[0] if hasattr(sv, "__len__") else sv

            st.session_state.results = {
                "prob": prob,
                "dossier_id": dossier_id or f"PB-{datetime.datetime.now().year}-ANONYME",
                "form": form_data,
                "shap_vals": shap_vals,
                "recap": {
                    "amount": float(loan_amount),
                    "months": int(loan_duration.split()[0]),
                    "income": float(income),
                },
            }

        res = st.session_state.results

        if res is not None:
            chip = f'<span class="pb-chip">{res["dossier_id"]}</span>'
        elif dossier_id:
            chip = f'<span class="pb-chip">{dossier_id}</span>'
        else:
            chip = f'<span class="pb-chip wait">{T[lang]["dossier_pending"]}</span>'

        with head_slot:
            html(f'<div class="pb-anchor"></div><div class="pb-ph"><div class="pb-ph-t">{T[lang]["panel_title"]}</div>{chip}</div>')

        with body_slot:
            html(result_html(res, lang) if res is not None else empty_html(lang))

    html(f'<div class="pb-foot">{T[lang]["final_disclaimer"]}</div>')

main_app()
