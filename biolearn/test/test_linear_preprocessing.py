import numpy as np
import pandas as pd
import pytest

from biolearn.data_library import GeoData
from biolearn.features import MissingFeaturesError, RequiredFeatures
from biolearn.model import LinearMethylationModel, LinearTranscriptomicModel
from biolearn.model_gallery import ModelGallery
from biolearn.util import get_data_file


def _linear_model(preprocess, model_class=LinearMethylationModel):
    coefficients = pd.DataFrame(
        {"CoefficientTraining": [0.5, 0.5, 1.0]},
        index=["cg1", "cg2", "intercept"],
    )
    return model_class(
        coefficients,
        transform=lambda x: x,
        preprocess=preprocess,
        name="TestClock",
    )


def _data(include_cg2=False):
    dnam = pd.DataFrame({"S1": [0.1]}, index=["cg1"])
    if include_cg2:
        dnam.loc["cg2"] = 0.2
    return GeoData(pd.DataFrame(index=dnam.columns), dnam=dnam)


def _fill_cg2(frame):
    return frame.reindex(["cg1", "cg2"], fill_value=0.2)


def test_linear_methylation_preprocessing_can_supply_missing_cpg():
    data = _data()
    original = data.dnam.copy()
    model = _linear_model(_fill_cg2)

    prediction = model.predict(data)

    assert prediction.loc["S1", "Predicted"] == pytest.approx(1.15)
    pd.testing.assert_frame_equal(data.dnam, original)


def test_linear_methylation_rejects_cpg_removed_by_preprocessing():
    data = _data(include_cg2=True)
    original = data.dnam.copy()
    model = _linear_model(lambda frame: frame.drop(index="cg2"))

    with pytest.raises(MissingFeaturesError, match=r"TestClock.*cg2"):
        model.predict(data)

    pd.testing.assert_frame_equal(data.dnam, original)


@pytest.mark.parametrize("missing_layer", [False, True])
def test_linear_methylation_rejects_inputs_still_missing(missing_layer):
    data = _data()
    if missing_layer:
        data.dnam = None
    model = _linear_model(lambda frame: frame)

    with pytest.raises(MissingFeaturesError, match=r"TestClock.*cg2"):
        model.predict(data)


@pytest.mark.parametrize("has_age", [False, True])
def test_processed_input_validation_preserves_metadata(has_age):
    class MetadataClock(LinearMethylationModel):
        def required_features(self):
            return RequiredFeatures("dnam", ("cg1", "cg2"), ("age",))

    data = _data()
    if has_age:
        data.metadata["age"] = 42
    original_metadata = data.metadata.copy()
    original_dnam = data.dnam.copy()
    model = _linear_model(_fill_cg2, model_class=MetadataClock)

    if has_age:
        prediction = model.predict(data)
        assert prediction.loc["S1", "Predicted"] == pytest.approx(1.15)
    else:
        with pytest.raises(
            MissingFeaturesError, match=r"Missing required metadata.*age"
        ):
            model.predict(data)

    pd.testing.assert_frame_equal(data.metadata, original_metadata)
    pd.testing.assert_frame_equal(data.dnam, original_dnam)


def test_linear_transcriptomic_model_still_tolerates_partial_input():
    data = _data()
    data.rna = data.dnam
    data.dnam = None
    model = _linear_model(
        lambda frame: frame.copy(), model_class=LinearTranscriptomicModel
    )

    prediction = model.predict(data)

    assert prediction.loc["S1", "Predicted"] == pytest.approx(1.05)


@pytest.mark.parametrize("imputation_method", [None, "dunedin"])
@pytest.mark.parametrize("missing_cpg", [False, True])
def test_dunedin_pace_preprocessing_preserves_predictions(
    imputation_method, missing_cpg
):
    gold = pd.read_csv(
        get_data_file("DunedinPACE_Gold_Means.csv"), index_col=0
    )["mean"]
    dnam = pd.DataFrame({"S1": gold, "S2": (gold + 0.01).clip(0, 1)})
    if missing_cpg:
        dnam = dnam.drop(index="cg00112187")
    original = dnam.copy()
    data = GeoData(pd.DataFrame(index=dnam.columns), dnam=dnam)
    model = ModelGallery().get(
        "DunedinPACE", imputation_method=imputation_method
    )

    prediction = model.predict(data)

    # Reference predictions from the accepted preprocessing path, before the
    # required_features validator was introduced. All inputs are synthetic.
    expected = [
        0.9911063319635292,
        0.992489206113851 if missing_cpg else 0.9911063319635292,
    ]
    assert prediction.index.tolist() == ["S1", "S2"]
    assert prediction.columns.tolist() == ["Predicted"]
    np.testing.assert_allclose(
        prediction["Predicted"], expected, rtol=1e-10, atol=1e-12
    )
    pd.testing.assert_frame_equal(data.dnam, original)
