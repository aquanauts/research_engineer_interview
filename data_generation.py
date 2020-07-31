import time
from dataclasses import dataclass, field
import datetime
import math

from typing import Dict, List, Iterator, Callable

import pandas as pd
import numpy as np

# ---------------------------------------------------------
# CODE IN THIS FILE SHOULD NOT BE MODIFIED BY THE CANDIDATE
# ---------------------------------------------------------

C_A_BETA_POWER = 4.0
C_B_BETA_POWER = 4.0
B_SCALE = 200.0


@dataclass(frozen=True)
class FeatureSetDateScope:
    date: datetime.date
    feature_set_name: str


@dataclass(frozen=True)
class DateRange:
    start_date: datetime.date
    end_date: datetime.date

    def get_dates(self) -> Iterator[datetime.date]:
        date = self.start_date
        while date <= self.end_date:
            yield date
            date += datetime.timedelta(days=1)


@dataclass(frozen=True)
class FeatureSetDateRangeScope(DateRange):
    feature_set_name: str


@dataclass(frozen=True)
class FeatureSetConfig:
    name: str
    num_features: int
    time_step_seconds: int
    data_generator: Callable[['VersionedDataSet', FeatureSetDateScope], pd.DataFrame]


@dataclass(frozen=True)
class DrivenFeatureSetConfig(FeatureSetConfig):
    driving_factors: int


@dataclass
class VersionedDataSet:
    data_version: int
    start_date: datetime.date
    end_date: datetime.date
    start_time: datetime.time
    end_time: datetime.time
    feature_set_a: DrivenFeatureSetConfig
    feature_set_b: DrivenFeatureSetConfig
    feature_set_c: FeatureSetConfig
    b_iterations: int
    feature_sets_by_name: Dict[str, FeatureSetConfig] = field(init=False)
    feature_set_name_to_index: Dict[str, int] = field(init=False)
    cov_a: np.ndarray = field(init=False)
    cov_b: np.ndarray = field(init=False)
    a_beta: np.ndarray = field(init=False)
    b_beta: np.ndarray = field(init=False)

    def __post_init__(self):
        feature_sets = [self.feature_set_a, self.feature_set_b, self.feature_set_c]
        self.feature_sets_by_name = {s.name: s for s in feature_sets}
        self.feature_set_name_to_index = {s.name: i for i, s in enumerate(feature_sets)}
        self.cov_a = feature_set_cov(self.feature_sets_by_name['A'])
        self.cov_b = feature_set_cov(self.feature_sets_by_name['B'])

        num_features = self.feature_set_c.num_features
        generator = make_generator(0)
        self.a_beta = generator.random([1, self.feature_set_a.num_features, num_features]) ** C_A_BETA_POWER
        self.b_beta = generator.random([1, self.feature_set_b.num_features, num_features]) ** C_B_BETA_POWER

    # ------- PUBLIC API, you will probably want to use these -------

    def dates(self) -> Iterator[datetime.date]:
        return DateRange(self.start_date, self.end_date).get_dates()

    def get_date_data(self, scope: FeatureSetDateScope) -> pd.DataFrame:
        if scope.date < self.start_date or scope.date > self.end_date:
            raise ValueError('Date out of range')

        config = self.feature_sets_by_name[scope.feature_set_name]
        data = config.data_generator(self, scope)

        index = self.make_time_index(config, scope)
        num_features = config.num_features
        columns = make_feature_names(scope.feature_set_name, num_features)

        df =  pd.DataFrame(index=index,
                            data=data,
                            columns=columns)
        order_generator = self.make_generator(scope)
        columns_reordered = np.array(columns)[np.argsort(order_generator.random(len(columns)))]
        return df.reindex(columns_reordered, axis=1)

    def get_date_range_data(self, scope: FeatureSetDateRangeScope) -> pd.DataFrame:
        return pd.concat([self.get_date_data(FeatureSetDateScope(date, scope.feature_set_name)) for date in scope.get_dates()])

    # ------- INTERNAL functions, you probably don't need these -------

    def seed(self, scope: FeatureSetDateScope) -> int:
        date_offset = (scope.date - self.start_date).days

        num_dates = (self.end_date - self.start_date).days
        return (self.data_version * num_dates + date_offset) * len(self.feature_set_name_to_index) \
            + self.feature_set_name_to_index[scope.feature_set_name]

    def make_generator(self, scope: FeatureSetDateScope) -> np.random.Generator:
        return make_generator(self.seed(scope))

    def samples_per_day(self, config: FeatureSetConfig) -> int:
        return int((datetime.datetime.combine(self.start_date, self.end_time)
                    - datetime.datetime.combine(self.start_date, self.start_time)).total_seconds()
                   / datetime.timedelta(seconds=config.time_step_seconds).total_seconds())

    def make_time_index(self, config: FeatureSetConfig, scope: FeatureSetDateScope) -> pd.DatetimeIndex:
        ref_time = pd.Timestamp(datetime.datetime.combine(scope.date, self.start_time))
        seconds_offset = np.arange(self.samples_per_day(config)) * config.time_step_seconds
        return pd.DatetimeIndex(ref_time + pd.to_timedelta(seconds_offset, unit='s'))

    def feature_set_A_generator(self, scope: FeatureSetDateScope) -> np.ndarray:
        return self.make_generator(scope).multivariate_normal(np.zeros([self.feature_set_a.num_features]),
                                                         self.cov_a,
                                                         size=self.samples_per_day(self.feature_set_a))

    def feature_set_B_generator(self, scope: FeatureSetDateScope) -> np.ndarray:
        generator = self.make_generator(scope)
        var_multiplier = np.zeros(self.feature_set_b.num_features)
        for _ in range(self.b_iterations):
            var_multiplier += generator.random(var_multiplier.shape)
            var_multiplier -= np.floor(var_multiplier)
        var_multiplier = (var_multiplier / 3.0) + 1.0
        adjusted_cov = self.cov_b * np.outer(var_multiplier, var_multiplier.T)

        return generator.multivariate_normal(
            np.zeros([self.feature_set_b.num_features]), adjusted_cov,
            size=self.samples_per_day(self.feature_set_b)) * B_SCALE

    def feature_set_C_generator(self, scope: FeatureSetDateScope) -> np.ndarray:
        generator = self.make_generator(scope)
        a_scope = FeatureSetDateScope(
            date=scope.date,
            feature_set_name='A')
        b_scope = FeatureSetDateScope(
            date=scope.date,
            feature_set_name='B')

        index = self.make_time_index(self.feature_set_c, scope)

        a_data = self.get_date_data(a_scope).reindex(index=index, method='ffill')
        a_data = a_data.reindex(sorted(a_data.columns), axis=1)
        b_data = self.get_date_data(b_scope).reindex(index=index, method='ffill')
        b_data = b_data.reindex(sorted(b_data.columns), axis=1)

        a_effect = np.matmul(a_data.values, self.a_beta).squeeze()
        b_effect = np.matmul(b_data.values, self.b_beta).squeeze()

        feature_effects = a_effect + b_effect

        return generator.normal(0, 1, feature_effects.shape) + feature_effects


def make_feature_names(set_name: str, num_features: int):
    return [set_name + '_f' + str(i).zfill(math.ceil(math.log10(num_features))) for i in range(num_features)]


def make_generator(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed=seed))


def feature_set_cov(config: DrivenFeatureSetConfig) -> np.ndarray:
    cov_generator = make_generator(0)
    cov_exp = cov_generator.normal(0, 1, [config.num_features, config.num_features])
    driving_cov_exp = cov_generator.normal(0, 1, [config.driving_factors, config.driving_factors])
    driving_cov = np.matmul(driving_cov_exp, driving_cov_exp.T)
    betas = cov_generator.normal(0, 1, [config.driving_factors, config.num_features])
    cov_exp += np.matmul(np.matmul(betas.T, driving_cov_exp), betas)
    cov = np.matmul(cov_exp, cov_exp.T)
    return cov


@dataclass
class FeatureSet:
    config: FeatureSetConfig
    value_generator: Callable[['FeatureSet', FeatureSetDateScope, np.random.Generator], np.ndarray]
    cov: np.ndarray = field(init=False)

    def __post_init__(self):
        self.cov = feature_set_cov(self.config)


DATA_VERSION_1 = VersionedDataSet(
    data_version=1,
    start_date=datetime.date(2010, 1, 1),
    end_date=datetime.date(2010, 2, 1),
    start_time=datetime.time(9, 0),
    end_time=datetime.time(15, 0),
    feature_set_a=DrivenFeatureSetConfig(name='A',
                                         num_features=15,
                                         time_step_seconds=60,
                                         driving_factors=10,
                                         data_generator=VersionedDataSet.feature_set_A_generator),
    feature_set_b=DrivenFeatureSetConfig(name='B',
                                         num_features=10,
                                         time_step_seconds=300,
                                         driving_factors=3,
                                         data_generator=VersionedDataSet.feature_set_B_generator),
    feature_set_c=FeatureSetConfig(name='C',
                                   num_features=5,
                                   time_step_seconds=60,
                                   data_generator=VersionedDataSet.feature_set_C_generator),
    b_iterations=1)

DATA_VERSION_2 = VersionedDataSet(
    data_version=2,
    start_date=datetime.date(2010, 1, 1),
    end_date=datetime.date(2011, 1, 1),
    start_time=datetime.time(3, 0),
    end_time=datetime.time(21, 0),
    feature_set_a=DrivenFeatureSetConfig(name='A',
                                         num_features=800,
                                         time_step_seconds=30,
                                         driving_factors=10,
                                         data_generator=VersionedDataSet.feature_set_A_generator),
    feature_set_b=DrivenFeatureSetConfig(name='B',
                                         num_features=10,
                                         time_step_seconds=300,
                                         driving_factors=3,
                                         data_generator=VersionedDataSet.feature_set_B_generator),
    feature_set_c=FeatureSetConfig(name='C',
                                   num_features=5,
                                   time_step_seconds=30,
                                   data_generator=VersionedDataSet.feature_set_C_generator),
    b_iterations=1_000_000)
