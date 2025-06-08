from datatime import date
from pydantic import BaseField, Field
from pydantic.functional_validators import AfterValidator
from typing import Annotated, Optional

def validate_and_clean_date(input_date: str) -> date:
    pass

def validate_full_name(input_name: str) -> str:
    pass

def validate_nationality(input_nationality: str) -> str:
    pass

class GuarantorInfo(BaseModel):
    name: str

class LetterOfGuarantee(BaseModel):
    date: Annotated[
        Optional[str],
        Field(description="Date on which this form is being filled up."),
        AfterValidator(validate_and_clean_date)
    ] = None

    name: Annotated[
        Optional[str],
        Field(description="Full name of the person filling the form."),
        AfterValidator(validate_full_name)
    ]
    nationality: Annotated[
        Optional[str],
        Field(description="Nationality of the person filling the form."),
        AfterValidator(validate_nationality)
    ]