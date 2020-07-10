# importing required libraries
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pickle

# initiate the app
app = dash.Dash()
server = app.server


# format the app
colors = {"background": "#111111", "text": "#009E73"}

app.layout = html.Div(
    children=[
        html.H1(
            "Fraud Detection", style={"textAlign": "center", "color": colors["text"]}
        ),
        # step
        html.Div(
            [
                html.H3("OTP Sent", style={"paddingRight": "30px"}),
                dcc.Input(
                    id="Step",
                    type="number",
                    min=0,
                    value=800,
                    style={"fontsize": 15, "width": 55, "color": colors["text"]},
                ),
            ],
            style={
                "display": "inline-block",
                "verticalAlign": "middle",
                "color": colors["text"],
            },
        ),
        # sender's original balance
        html.Div(
            [
                html.H3("Sender's Orig Balance", style={"paddingRight": "30px"}),
                dcc.Input(
                    id="oldbalanceOrg",
                    type="number",
                    min=0,
                    value=10000,
                    style={"fontsize": 15, "width": 55, "color": colors["text"]},
                ),
            ],
            style={
                "display": "inline-block",
                "verticalAlign": "middle",
                "color": colors["text"],
            },
        ),
        # new balance origin
        html.Div(
            [
                html.H3("Sender's New Balance", style={"paddingRight": "30px"}),
                dcc.Input(
                    id="newbalanceOrig",
                    type="number",
                    min=0,
                    value=9000,
                    style={"fontsize": 15, "width": 55, "color": colors["text"]},
                ),
            ],
            style={
                "display": "inline-block",
                "verticalAlign": "middle",
                "color": colors["text"],
            },
        ),
        # old balance destination
        html.Div(
            [
                html.H3("Receiver's Orig Balance", style={"paddingRight": "30px"}),
                dcc.Input(
                    id="oldbalanceDest",
                    type="number",
                    min=0,
                    value=10000,
                    style={"fontsize": 15, "width": 55, "color": colors["text"]},
                ),
            ],
            style={
                "display": "inline-block",
                "verticalAlign": "middle",
                "color": colors["text"],
            },
        ),
        # new balance destination
        html.Div(
            [
                html.H3("Sender's New Balance", style={"paddingRight": "30px"}),
                dcc.Input(
                    id="newbalanceDest",
                    type="number",
                    min=0,
                    value=11000,
                    style={"fontsize": 15, "width": 55, "color": colors["text"]},
                ),
            ],
            style={
                "display": "inline-block",
                "verticalAlign": "middle",
                "color": colors["text"],
            },
        ),
        # recognized device
        html.Div(
            [
                html.H3("Unrecognized Device", style={"paddingRight": "30px"}),
                dcc.Dropdown(
                    id="isUnrecognizedDevice",
                    options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                    value=1,
                    clearable=False,
                    style={"fontsize": 15, "width": 55, "color": colors["text"]},
                ),
            ],
            style={
                "width": "100%",
                "display": "flex",
                "align-items": "center",
                "justify-content": "center",
                "color": colors["text"],
            },
        ),
        # recognized location
        html.Div(
            [
                html.H3("Unrecongnized Location: ", style={"paddingRight": "30px"}),
                dcc.Dropdown(
                    id="isOutsideLocation",
                    options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                    value=1,
                    clearable=False,
                    style={"fontsize": 15, "width": 55, "color": colors["text"]},
                ),
            ],
            style={
                "width": "100%",
                "display": "flex",
                "align-items": "center",
                "justify-content": "center",
                "color": colors["text"],
            },
        ),
        # the submit button
        html.Div(
            [
                html.Button(
                    id="submit-button",
                    children="Find",
                    n_clicks=0,
                    style={
                        "fontSize": 20,
                        "marginLeft": "20px",
                        "color": colors["text"],
                    },
                )
            ],
            style={
                "width": "100%",
                "display": "flex",
                "align-items": "center",
                "justify-content": "center",
                "color": colors["text"],
            },
        ),
        # the graphs
        dcc.Graph(id="Verdict"),
        dcc.Graph(id="Model_Evaluation"),
    ],
    style={
        "display": "inline-block",
        "verticalAlign": "middle",
        "backgroundColor": colors["background"],
    },
)

# app functions
@app.callback(
    [
        Output(component_id="Verdict", component_property="figure"),
        Output(component_id="Model_Evaluation", component_property="figure"),
    ],
    [Input("submit-button", "n_clicks")],
    [
        State("Step", "value"),
        State("oldbalanceOrg", "value"),
        State("newbalanceOrig", "value"),
        State("oldbalanceDest", "value"),
        State("newbalanceDest", "value"),
        State("isUnrecognizedDevice", "value"),
        State("isOutsideLocation", "value"),
    ],
)

# start the function
def Fraud_Verdict(
    n_clicks,
    Step,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest,
    isUnrecognizedDevice,
    isOutsideLocation,
):
    # test case plot
    test_case = [
        Step,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest,
        isUnrecognizedDevice,
        isOutsideLocation,
    ]
    test_case = pd.DataFrame(test_case).T
    test_case.columns = [
        "step",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isUnrecognizedDevice",
        "isOutsideLocation",
    ]
    loaded_model = pickle.load(open("fraud_detection.pkl", "rb"))
    test_case_verdict = loaded_model.predict(test_case)
    verdict = np.where(
        test_case_verdict == 0, "Regular Transaction", "Suspicious Transaction"
    )[0]
    testcase_prob = loaded_model.predict_proba(test_case).tolist()[0]

    verdict_plot = go.Bar(x=["Regular Case", "Fraud Case"], y=testcase_prob)
    verdict_layout = go.Layout(
        title="Verdict: " + verdict,
        xaxis=dict(title="Case Category"),
        yaxis=dict(title="Probability"),
    )
    verdict_fig = go.Figure(data=[verdict_plot], layout=verdict_layout)
    verdict_fig.update_layout(
        plot_bgcolor=colors["background"],
        font=dict(color=colors["text"]),
        paper_bgcolor=colors["background"],
    )

    # model evaluation table
    evaluation = pd.read_csv("report.csv")
    evaluation = evaluation.rename(columns={"Unnamed: 0": "Category"})
    evaluation.iloc[:, 1:] = evaluation.iloc[:, 1:].round(5)

    eval_table = go.Table(
        header=dict(
            values=list(evaluation.columns), fill_color="paleturquoise", align="left"
        ),
        cells=dict(
            values=[
                evaluation["Category"],
                evaluation["precision"],
                evaluation["recall"],
                evaluation["f1-score"],
                evaluation["support"],
            ],
            fill_color="lavender",
            align="left",
        ),
    )

    eval_layout = go.Layout(
        title="Model Evaluaiton", xaxis=dict(title=""), yaxis=dict(title=""),
    )

    eval_fig = go.Figure(data=[eval_table], layout=eval_layout)
    eval_fig.update_layout(
        plot_bgcolor=colors["background"],
        font=dict(color=colors["text"]),
        paper_bgcolor=colors["background"],
    )

    return (verdict_fig, eval_fig)

# launch the app
if __name__ == "__main__":
    app.run_server(debug=False, threaded=False)