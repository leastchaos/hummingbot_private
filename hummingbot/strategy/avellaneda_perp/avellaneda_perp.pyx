import datetime
import logging
import os
import time
from decimal import Decimal
from math import ceil, floor, isnan
from typing import Dict, List, Tuple, Union
from enum import Enum
from itertools import chain
import numpy as np
import pandas as pd

from hummingbot.connector.exchange_base import ExchangeBase
from hummingbot.connector.exchange_base cimport ExchangeBase
from hummingbot.core.clock cimport Clock
from hummingbot.core.data_type.order_candidate import PerpetualOrderCandidate

from hummingbot.client.config.config_helpers import ClientConfigAdapter
from hummingbot.core.data_type.common import (
    OrderType,
    PriceType,
    TradeType,
    PositionMode,
    PositionSide,
    PositionAction
)
from hummingbot.core.utils.estimate_fee import build_perpetual_trade_fee
from hummingbot.core.data_type.limit_order cimport LimitOrder
from hummingbot.core.data_type.limit_order import LimitOrder
from hummingbot.core.network_iterator import NetworkStatus
from hummingbot.core.utils import map_df_to_str
from hummingbot.strategy.__utils__.trailing_indicators.instant_volatility import InstantVolatilityIndicator
from hummingbot.strategy.__utils__.trailing_indicators.trading_intensity import TradingIntensityIndicator
from hummingbot.strategy.avellaneda_perp.avellaneda_perp_config_map_pydantic import (
    AvellanedaPerpConfigMap,
    DailyBetweenTimesModel,
    FromDateToDateModel,
    MultiOrderLevelModel,
    TrackHangingOrdersModel,
)
from hummingbot.strategy.conditional_execution_state import (
    RunAlwaysExecutionState,
    RunInTimeConditionalExecutionState
)
from hummingbot.strategy.data_types import (
    PriceSize,
    Proposal,
)
from hummingbot.strategy.hanging_orders_tracker import (
    CreatedPairOfOrders,
    HangingOrdersTracker,
)
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.order_book_asset_price_delegate import OrderBookAssetPriceDelegate
from hummingbot.strategy.order_tracker cimport OrderTracker
from hummingbot.strategy.strategy_base import StrategyBase
from hummingbot.strategy.utils import order_age

NaN = float("nan")
s_decimal_zero = Decimal(0)
s_decimal_neg_one = Decimal(-1)
s_decimal_one = Decimal(1)
pmm_logger = None

class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"

cdef class AvellanedaPerpStrategy(StrategyBase):
    OPTION_LOG_CREATE_ORDER = 1 << 3
    OPTION_LOG_MAKER_ORDER_FILLED = 1 << 4
    OPTION_LOG_STATUS_REPORT = 1 << 5
    OPTION_LOG_ALL = 0x7fffffffffffffff

    @classmethod
    def logger(cls):
        global pmm_logger
        if pmm_logger is None:
            pmm_logger = logging.getLogger(__name__)
        return pmm_logger

    def init_params(self,
                    config_map: Union[AvellanedaPerpConfigMap, ClientConfigAdapter],
                    market_info: MarketTradingPairTuple,
                    logging_options: int = OPTION_LOG_ALL,
                    status_report_interval: float = 900,
                    hb_app_notification: bool = False,
                    debug_csv_path: str = '',
                    is_debug: bool = False,
                    ):
        self._sb_order_tracker = OrderTracker()
        self._config_map = config_map
        if self.direction == Direction.BOTH and self.position_mode != PositionMode.ONEWAY:
            raise ValueError("Direction BOTH can only be used with oneway position mode.")
        self._market_info = market_info
        self._price_delegate = OrderBookAssetPriceDelegate(market_info.market, market_info.trading_pair)
        self._hb_app_notification = hb_app_notification
        self._hanging_orders_enabled = False
        self._hanging_orders_cancel_pct = Decimal("10")
        self._hanging_orders_tracker = HangingOrdersTracker(self,
                                                            self._hanging_orders_cancel_pct / Decimal('100'))

        self._cancel_timestamp = 0
        self._create_timestamp = 0
        self._limit_order_type = self._market_info.market.get_maker_order_type()
        self._all_markets_ready = False
        self._filled_buys_balance = 0
        self._filled_sells_balance = 0
        self._logging_options = logging_options
        self._last_timestamp = 0
        self._status_report_interval = status_report_interval
        self._last_own_trade_price = Decimal('nan')

        self.c_add_markets([market_info.market])
        self._volatility_buffer_size = 0
        self._trading_intensity_buffer_size = 0
        self._ticks_to_be_ready = -1
        self._avg_vol = None
        self._trading_intensity = None
        self._last_sampling_timestamp = 0
        self._alpha = None
        self._kappa = None
        self._execution_mode = None
        self._execution_timeframe = None
        self._execution_state = None
        self._start_time = None
        self._end_time = None
        self._reservation_price = s_decimal_zero
        self._optimal_spread = s_decimal_zero
        self._optimal_ask = s_decimal_zero
        self._optimal_bid = s_decimal_zero
        self._debug_csv_path = debug_csv_path
        self._is_debug = is_debug
        try:
            if self._is_debug:
                os.unlink(self._debug_csv_path)
        except FileNotFoundError:
            pass

        self.get_config_map_execution_mode()
        self.get_config_map_hanging_orders()

    def all_markets_ready(self):
        return all([market.ready for market in self._sb_markets])

    @property
    def min_spread(self):
        return self._config_map.min_spread

    @property
    def avg_vol(self):
        return self._avg_vol

    @property
    def direction(self) -> Direction:
        if self._config_map.direction == "LONG":
            return Direction.LONG
        elif self._config_map.direction == "SHORT":
            return Direction.SHORT
        elif self._config_map.direction == "BOTH":
            return Direction.BOTH

    @avg_vol.setter
    def avg_vol(self, indicator: InstantVolatilityIndicator):
        self._avg_vol = indicator

    @property
    def trading_intensity(self):
        return self._trading_intensity

    @trading_intensity.setter
    def trading_intensity(self, indicator: TradingIntensityIndicator):
        self._trading_intensity = indicator

    @property
    def market_info(self) -> MarketTradingPairTuple:
        return self._market_info

    @property
    def order_refresh_tolerance_pct(self) -> Decimal:
        return self._config_map.order_refresh_tolerance_pct

    @property
    def order_refresh_tolerance(self) -> Decimal:
        return self._config_map.order_refresh_tolerance_pct / Decimal('100')

    @property
    def order_amount(self) -> Decimal:
        return self._config_map.order_amount

    @property
    def inventory_target_base_pct(self) -> Decimal:
        return self._config_map.inventory_target_base_pct

    @property
    def inventory_target_base(self) -> Decimal:
        return self.inventory_target_base_pct / Decimal('100')

    @inventory_target_base.setter
    def inventory_target_base(self, value: Decimal):
        self.inventory_target_base_pct = value * Decimal('100')

    @property
    def order_optimization_enabled(self) -> bool:
        return self._config_map.order_optimization_enabled

    @property
    def order_refresh_time(self) -> float:
        return self._config_map.order_refresh_time

    @property
    def filled_order_delay(self) -> float:
        return self._config_map.filled_order_delay

    @property
    def order_override(self) -> Dict[str, any]:
        return self._config_map.order_override

    @property
    def order_levels(self) -> int:
        if self._config_map.order_levels_mode.title == MultiOrderLevelModel.Config.title:
            return self._config_map.order_levels_mode.order_levels
        else:
            return 0

    @property
    def level_distances(self) -> int:
        if self._config_map.order_levels_mode.title == MultiOrderLevelModel.Config.title:
            return self._config_map.order_levels_mode.level_distances
        else:
            return 0

    @property
    def max_order_age(self):
        return self._config_map.max_order_age

    @property
    def add_transaction_costs_to_orders(self) -> bool:
        return self._config_map.add_transaction_costs

    @property
    def base_asset(self):
        return self._market_info.base_asset

    @property
    def quote_asset(self):
        return self._market_info.quote_asset

    @property
    def trading_pair(self):
        return self._market_info.trading_pair

    @property
    def gamma(self) -> Decimal:
        return self._config_map.risk_factor

    @property
    def alpha(self) -> Decimal:
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def kappa(self) -> Decimal:
        return self._kappa

    @kappa.setter
    def kappa(self, value):
        self._kappa = value

    @property
    def eta(self) -> Decimal:
        return self._config_map.order_amount_shape_factor

    @property
    def reservation_price(self) -> Decimal:
        return self._reservation_price

    @reservation_price.setter
    def reservation_price(self, value):
        self._reservation_price = value

    @property
    def optimal_spread(self) -> Decimal:
        return self._optimal_spread

    @property
    def optimal_ask(self) -> Decimal:
        return self._optimal_ask

    @optimal_ask.setter
    def optimal_ask(self, value):
        self._optimal_ask = value

    @property
    def optimal_bid(self) -> Decimal:
        return self._optimal_bid

    @optimal_bid.setter
    def optimal_bid(self, value):
        self._optimal_bid = value

    @property
    def execution_timeframe(self):
        return self._execution_timeframe

    @property
    def start_time(self) -> time:
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        self._start_time = value

    @property
    def end_time(self) -> time:
        return self._end_time

    @end_time.setter
    def end_time(self, value):
        self._end_time = value

    @property
    def leverage(self) -> int:
        return self._config_map.leverage
    
    @property
    def position_mode(self) -> PositionMode:
        if self._config_map.position_mode == "HEDGE":
            return PositionMode.HEDGE
        elif self._config_map.position_mode == "ONEWAY":
            return PositionMode.ONEWAY
        raise ValueError(f"Invalid position mode {self._config_map.position_mode}")

    def get_price(self) -> Decimal:
        return self.get_mid_price()

    def get_last_price(self) -> Decimal:
        return self._market_info.get_last_price()

    def get_mid_price(self) -> Decimal:
        return self.c_get_mid_price()

    cdef object c_get_mid_price(self):
        return self._price_delegate.get_price_by_type(PriceType.MidPrice)

    @property
    def market_info_to_active_orders(self) -> Dict[MarketTradingPairTuple, List[LimitOrder]]:
        return self._sb_order_tracker.market_pair_to_active_orders

    @property
    def active_orders(self) -> List[LimitOrder]:
        if self._market_info not in self.market_info_to_active_orders:
            return []
        return self.market_info_to_active_orders[self._market_info]

    @property
    def active_non_hanging_orders(self) -> List[LimitOrder]:
        orders = [o for o in self.active_orders if not self._hanging_orders_tracker.is_order_id_in_hanging_orders(o.client_order_id)]
        return orders

    @property
    def active_buys(self) -> List[LimitOrder]:
        return [o for o in self.active_orders if o.is_buy]

    @property
    def active_sells(self) -> List[LimitOrder]:
        return [o for o in self.active_orders if not o.is_buy]

    @property
    def logging_options(self) -> int:
        return self._logging_options

    @logging_options.setter
    def logging_options(self, int64_t logging_options):
        self._logging_options = logging_options

    @property
    def hanging_orders_tracker(self):
        return self._hanging_orders_tracker

    @property
    def exchange_name(self):
        return self._config_map.exchange

    def update_from_config_map(self):
        self.get_config_map_execution_mode()
        self.get_config_map_hanging_orders()
        self.get_config_map_indicators()

    def get_config_map_execution_mode(self):
        try:
            execution_mode = self._config_map.execution_timeframe_mode.title
            execution_timeframe = self._config_map.execution_timeframe_mode.Config.title
            if execution_mode == FromDateToDateModel.Config.title:
                start_time = self._config_map.execution_timeframe_mode.start_datetime
                end_time = self._config_map.execution_timeframe_mode.end_datetime
                execution_state = RunInTimeConditionalExecutionState(start_timestamp=start_time, end_timestamp=end_time)
            elif execution_mode == DailyBetweenTimesModel.Config.title:
                start_time = self._config_map.execution_timeframe_mode.start_time
                end_time = self._config_map.execution_timeframe_mode.end_time
                execution_state = RunInTimeConditionalExecutionState(start_timestamp=start_time, end_timestamp=end_time)
            else:
                start_time = None
                end_time = None
                execution_state = RunAlwaysExecutionState()

            # Something has changed?
            if self._execution_state is None or self._execution_state != execution_state:
                self._execution_state = execution_state
                self._execution_mode = execution_mode
                self._execution_timeframe = execution_timeframe
                self._start_time = start_time
                self._end_time = end_time
        except AttributeError:
            # A parameter is missing in the execution timeframe mode configuration
            pass

    def get_config_map_hanging_orders(self):
        if self._config_map.hanging_orders_mode.title == TrackHangingOrdersModel.Config.title:
            hanging_orders_enabled = True
            hanging_orders_cancel_pct = self._config_map.hanging_orders_mode.hanging_orders_cancel_pct
        else:
            hanging_orders_enabled = False
            hanging_orders_cancel_pct = Decimal("0")

        if self._hanging_orders_enabled != hanging_orders_enabled:
            # Hanging order tracker instance doesn't exist - create from scratch
            self._hanging_orders_enabled = hanging_orders_enabled
            self._hanging_orders_cancel_pct = hanging_orders_cancel_pct
            self._hanging_orders_tracker = HangingOrdersTracker(self,
                                                                hanging_orders_cancel_pct / Decimal('100'))
            self._hanging_orders_tracker.register_events(self.active_markets)
        elif self._hanging_orders_cancel_pct != hanging_orders_cancel_pct:
            # Hanging order tracker instance existst - only update variable
            self._hanging_orders_cancel_pct = hanging_orders_cancel_pct
            self._hanging_orders_tracker.hanging_orders_cancel_pct = hanging_orders_cancel_pct / Decimal('100')

    def get_config_map_indicators(self):
        volatility_buffer_size = self._config_map.volatility_buffer_size
        trading_intensity_buffer_size = self._config_map.trading_intensity_buffer_size
        ticks_to_be_ready_after = max(volatility_buffer_size, trading_intensity_buffer_size)
        ticks_to_be_ready_before = max(self._volatility_buffer_size, self._trading_intensity_buffer_size)

        if self._volatility_buffer_size == 0 or self._volatility_buffer_size != volatility_buffer_size:
            self._volatility_buffer_size = volatility_buffer_size

            if self._avg_vol is None:
                self._avg_vol = InstantVolatilityIndicator(sampling_length=volatility_buffer_size)
            else:
                self._avg_vol.sampling_length = volatility_buffer_size

        if (
            self._trading_intensity_buffer_size == 0
            or self._trading_intensity_buffer_size != trading_intensity_buffer_size
        ):
            self._trading_intensity_buffer_size = trading_intensity_buffer_size
            if self._trading_intensity is not None:
                self._trading_intensity.sampling_length = trading_intensity_buffer_size

        if self._trading_intensity is None and self.market_info.market.ready:
            self._trading_intensity = TradingIntensityIndicator(
                order_book=self.market_info.order_book,
                price_delegate=self._price_delegate,
                sampling_length=self._trading_intensity_buffer_size,
            )

        self._ticks_to_be_ready += (ticks_to_be_ready_after - ticks_to_be_ready_before)
        if self._ticks_to_be_ready < 0:
            self._ticks_to_be_ready = 0

    def get_base_available_amount(self) -> Decimal:
        """
        Get the value of the derivative base asset.
        :param market_pair: The market pair to get the value of the derivative base asset.
        :return: The value of the derivative base asset.
        """
        # for future, available amount to sell depends on quote balance
        return self.get_quote_available_amount() / self.get_price()

    def get_quote_available_amount(self) -> Decimal:
        return self.market_info.market.get_available_balance(self.market_info.quote_asset) * self.leverage


    def get_base_amount(self) -> Decimal:
        trading_pair = self.trading_pair
        positions: List[Position] = [
            position
            for position in self.market_info.market.account_positions.values()
            if not isinstance(position, PositionMode) and position.trading_pair == trading_pair
        ]
        if self.direction == Direction.SHORT:
            both_sum = sum([
                position.amount
                for position in positions
                if position.position_side == PositionSide.BOTH])
            short_sum = - sum([
                    position.amount
                    for position in positions
                    if position.position_side == PositionSide.SHORT
                ])
            return both_sum + short_sum
        return sum([
            position.amount
            for position in positions
            if position.position_side == PositionSide.LONG or position.position_side == PositionSide.BOTH
        ])


    def get_quote_amount(self) -> Decimal:
        return self.market_info.market.get_balance(self.market_info.quote_asset) * self.leverage
    
    def pure_mm_assets_df(self, to_show_current_pct: bool) -> pd.DataFrame:
        market, trading_pair, base_asset, quote_asset = self._market_info
        price = self._price_delegate.get_price_by_type(PriceType.MidPrice)
        base_balance = float(self.get_base_amount())
        quote_balance = float(self.get_quote_amount())
        available_base_balance = float(self.get_base_available_amount())
        available_quote_balance = float(self.get_quote_available_amount())
        base_value = base_balance * float(price)
        total_in_quote = base_value + quote_balance
        base_ratio = base_value / total_in_quote if total_in_quote > 0 else 0
        quote_ratio = quote_balance / total_in_quote if total_in_quote > 0 else 0
        data = [
            ["", base_asset, quote_asset],
            ["Total Balance", round(base_balance, 4), round(quote_balance, 4)],
            ["Available Balance", round(available_base_balance, 4), round(available_quote_balance, 4)],
            [f"Current Value ({quote_asset})", round(base_value, 4), round(quote_balance, 4)]
        ]
        if to_show_current_pct:
            data.append(["Current %", f"{base_ratio:.1%}", f"{quote_ratio:.1%}"])
        df = pd.DataFrame(data=data)
        return df

    def active_orders_df(self) -> pd.DataFrame:
        market, trading_pair, base_asset, quote_asset = self._market_info
        price = self.get_price()
        active_orders = self.active_orders
        no_sells = len([o for o in active_orders if not o.is_buy and o.client_order_id and
                        not self._hanging_orders_tracker.is_order_id_in_hanging_orders(o.client_order_id)])
        active_orders.sort(key=lambda x: x.price, reverse=True)
        columns = ["Level", "Type", "Price", "Spread", "Amount (Orig)", "Amount (Adj)", "Age"]
        data = []
        lvl_buy, lvl_sell = 0, 0
        for idx in range(0, len(active_orders)):
            order = active_orders[idx]
            is_hanging_order = self._hanging_orders_tracker.is_order_id_in_hanging_orders(order.client_order_id)
            if not is_hanging_order:
                if order.is_buy:
                    level = lvl_buy + 1
                    lvl_buy += 1
                else:
                    level = no_sells - lvl_sell
                    lvl_sell += 1
            spread = 0 if price == 0 else abs(order.price - price) / price
            age = pd.Timestamp(order_age(order, self._current_timestamp), unit='s').strftime('%H:%M:%S')

            amount_orig = self._config_map.order_amount
            if is_hanging_order:
                amount_orig = order.quantity
                level = "hang"
            data.append([
                level,
                "buy" if order.is_buy else "sell",
                float(order.price),
                f"{spread:.2%}",
                float(amount_orig),
                float(order.quantity),
                age
            ])

        return pd.DataFrame(data=data, columns=columns)

    def market_status_data_frame(self, market_trading_pair_tuples: List[MarketTradingPairTuple]) -> pd.DataFrame:
        markets_data = []
        markets_columns = ["Exchange", "Market", "Best Bid", "Best Ask", f"MidPrice"]
        markets_columns.append('Reservation Price')
        markets_columns.append('Optimal Spread')
        market_books = [(self._market_info.market, self._market_info.trading_pair)]
        for market, trading_pair in market_books:
            bid_price = market.get_price(trading_pair, False)
            ask_price = market.get_price(trading_pair, True)
            ref_price = self.get_price()
            markets_data.append([
                market.display_name,
                trading_pair,
                float(bid_price),
                float(ask_price),
                float(ref_price),
                round(self._reservation_price, 5),
                round(self._optimal_spread, 5),
            ])
        return pd.DataFrame(data=markets_data, columns=markets_columns).replace(np.nan, '', regex=True)
    def get_positions(self) -> List[Position]:
        """
        Get the active positions of a market.
        :param market_pair: Market pair to get the positions of.
        :return: The active positions of the market.
        """
        trading_pair = self.trading_pair
        positions: List[Position] = [
            position
            for position in self.market_info.market.account_positions.values()
            if not isinstance(position, PositionMode) and position.trading_pair == trading_pair
        ]
        return positions

    def active_positions_df(self) -> pd.DataFrame:
        """
        Get the active positions of all markets.
        :return: The active positions of all markets.
        """
        columns = ["Symbol", "Type", "Entry", "Amount", "Leverage"]
        data = []
        for position in self.get_positions():
            if not position:
                continue
            data.append(
                [
                    position.trading_pair,
                    position.position_side.name,
                    position.entry_price,
                    position.amount,
                    position.leverage,
                ]
            )
        return pd.DataFrame(data=data, columns=columns)

    def format_status(self) -> str:
        if not self._all_markets_ready:
            return "Market connectors are not ready."
        cdef:
            list lines = []
            list warning_lines = []
        warning_lines.extend(self.network_warning([self._market_info]))

        markets_df = self.market_status_data_frame([self._market_info])
        lines.extend(["", "  Markets:"] + ["    " + line for line in markets_df.to_string(index=False).split("\n")])

        assets_df = map_df_to_str(self.pure_mm_assets_df(True))
        first_col_length = max(*assets_df[0].apply(len))
        df_lines = assets_df.to_string(index=False, header=False,
                                       formatters={0: ("{:<" + str(first_col_length) + "}").format}).split("\n")
        lines.extend(["", "  Assets:"] + ["    " + line for line in df_lines])

        # See if there are any open orders.
        if len(self.active_orders) > 0:
            df = self.active_orders_df()
            lines.extend(["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        else:
            lines.extend(["", "  No active maker orders."])

        # See if there are any open positions.
        if len(self.get_positions()) > 0:
            df = self.active_positions_df()
            lines.extend(["", "  Positions:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        else:
            lines.extend(["", "  No active positions."])
        volatility_pct = self._avg_vol.current_value / float(self.get_price()) * 100.0
        if all((self.gamma, self._alpha, self._kappa, not isnan(volatility_pct))):
            lines.extend(["", f"  Strategy parameters:",
                          f"    q = {self.c_get_q_ratio():.3f}",
                          f"    risk_factor(\u03B3)= {self.gamma:.5E}",
                          f"    order_book_intensity_factor(\u0391)= {self._alpha:.5E}",
                          f"    order_book_depth_factor(\u03BA)= {self._kappa:.5E}",
                          f"    volatility= {volatility_pct:.3f}%"]
                          )
            if self._execution_state.time_left is not None:
                lines.extend([f"    time until end of trading cycle = {str(datetime.timedelta(seconds=float(self._execution_state.time_left)//1e3))}"])
            else:
                lines.extend([f"    time until end of trading cycle = N/A"])

        warning_lines.extend(self.balance_warning([self._market_info]))

        if len(warning_lines) > 0:
            lines.extend(["", "*** WARNINGS ***"] + warning_lines)

        return "\n".join(lines)

    def execute_orders_proposal(self, proposal: Proposal):
        return self.c_execute_orders_proposal(proposal)

    def cancel_order(self, order_id: str):
        return self.c_cancel_order(self._market_info, order_id)

    cdef c_start(self, Clock clock, double timestamp):
        StrategyBase.c_start(self, clock, timestamp)
        self.update_from_config_map()
        self._last_timestamp = timestamp

        self._hanging_orders_tracker.register_events(self.active_markets)

        if self._hanging_orders_enabled:
            # start tracking any restored limit order
            restored_order_ids = self.c_track_restored_orders(self.market_info)
            for order_id in restored_order_ids:
                order = next(o for o in self.market_info.market.limit_orders if o.client_order_id == order_id)
                if order:
                    self._hanging_orders_tracker.add_as_hanging_order(order)

        self._execution_state.time_left = self._execution_state.closing_time
        self.market_info.market.set_leverage(self.trading_pair, self.leverage)
        self.logger().info("Setting position mode to %s" % self.position_mode)
        self.market_info.market.set_position_mode(self.position_mode)

    def start(self, clock: Clock, timestamp: float):
        self.c_start(clock, timestamp)

    cdef c_stop(self, Clock clock):
        self._hanging_orders_tracker.unregister_events(self.active_markets)
        StrategyBase.c_stop(self, clock)

    cdef c_tick(self, double timestamp):
        StrategyBase.c_tick(self, timestamp)
        cdef:
            int64_t current_tick = <int64_t>(timestamp // self._status_report_interval)
            int64_t last_tick = <int64_t>(self._last_timestamp // self._status_report_interval)
            bint should_report_warnings = ((current_tick > last_tick) and
                                           (self._logging_options & self.OPTION_LOG_STATUS_REPORT))
            object proposal

        try:
            if not self._all_markets_ready:
                self._all_markets_ready = all([mkt.ready for mkt in self._sb_markets])
                if not self._all_markets_ready:
                    # Markets not ready yet. Don't do anything.
                    if should_report_warnings:
                        self.logger().warning(f"Markets are not ready. No market making trades are permitted.")
                    return

            if should_report_warnings:
                if not all([mkt.network_status is NetworkStatus.CONNECTED for mkt in self._sb_markets]):
                    self.logger().warning(f"WARNING: Some markets are not connected or are down at the moment. Market "
                                          f"making may be dangerous when markets or networks are unstable.")

            # Updates settings from config map if changed
            self.update_from_config_map()

            self.c_collect_market_variables(timestamp)

            if self.c_is_algorithm_ready():
                if self._create_timestamp <= self._current_timestamp:
                    # Measure order book liquidity
                    self.c_measure_order_book_liquidity()

                self._hanging_orders_tracker.process_tick()

                # Needs to be executed at all times to not to have active order leftovers after a trading session ends
                self.c_cancel_active_orders_on_max_age_limit()

                # process_tick() is only called if within a trading timeframe
                self._execution_state.process_tick(timestamp, self)

            else:
                # Only if snapshots are different - for trading intensity - a market order happened
                if self.c_is_algorithm_changed():
                    self._ticks_to_be_ready -= 1
                    if self._ticks_to_be_ready % 5 == 0:
                        self.logger().info(f"Calculating volatility, estimating order book liquidity ... {self._ticks_to_be_ready} ticks to fill buffers")
                else:
                    self.logger().info(f"Calculating volatility, estimating order book liquidity ... no trades tick")
        finally:
            self._last_timestamp = timestamp

    def process_tick(self, timestamp: float):
        proposal = None
        # Trading is allowed
        if self._create_timestamp <= self._current_timestamp:
            # 1. Calculate reservation price and optimal spread from gamma, alpha, kappa and volatility
            self.c_calculate_reservation_price_and_optimal_spread()
            # 2. Check if calculated prices make sense
            if self._optimal_bid > 0 and self._optimal_ask > 0:
                # 3. Create base order proposals
                proposal = self.c_create_base_proposal()
                # 4. Apply functions that modify orders amount
                self.c_apply_order_amount_eta_transformation(proposal)
                # 5. Apply functions that modify orders price
                self.c_apply_order_price_modifiers(proposal)
                # 6. Apply budget constraint, i.e. can't buy/sell more than what you have.
                self.apply_budget_constraint(proposal)

                self.c_cancel_active_orders(proposal)

        if self.c_to_create_orders(proposal):
            self.c_execute_orders_proposal(proposal)

        if self._is_debug:
            self.dump_debug_variables()

    cdef c_collect_market_variables(self, double timestamp):
        market, trading_pair, base_asset, quote_asset = self._market_info
        self._last_sampling_timestamp = timestamp

        price = self.get_price()
        self._avg_vol.add_sample(price)
        self._trading_intensity.calculate(timestamp)

    def collect_market_variables(self, timestamp: float):
        self.c_collect_market_variables(timestamp)

    cdef double c_get_spread(self):
        cdef:
            ExchangeBase market = self._market_info.market
            str trading_pair = self._market_info.trading_pair

        return market.c_get_price(trading_pair, True) - market.c_get_price(trading_pair, False)

    def get_spread(self) -> float:
        return self.c_get_spread()

    def get_volatility(self) -> Decimal:
        return Decimal(str(self._avg_vol.current_value))

    cdef c_measure_order_book_liquidity(self):

        self._alpha, self._kappa = self._trading_intensity.current_value

        self._alpha = Decimal(self._alpha)
        self._kappa = Decimal(self._kappa)

        if self._is_debug:
            self.logger().info(f"alpha={self._alpha:.4f} | "
                               f"kappa={self._kappa:.4f}")

    def measure_order_book_liquidity(self):
        return self.c_measure_order_book_liquidity()

    cdef object c_get_q_ratio(self):
        # The amount of stocks owned - q - has to be in relative units, not absolute, because changing the portfolio size shouldn't change the reservation price
        # The reservation price should concern itself only with the strategy performance, i.e. amount of stocks relative to the target
        inventory = Decimal(str(self.c_calculate_inventory()))
        if inventory == 0:
            return

        q_target = Decimal(str(self.c_calculate_target_inventory()))
        q = (self.get_base_amount() - q_target) / (inventory)
        return q

    cdef c_calculate_reservation_price_and_optimal_spread(self):
        cdef:
            ExchangeBase market = self._market_info.market

        # Current mid price
        price = self.get_price()
        q = self.c_get_q_ratio()
        # Volatility has to be in absolute values (prices) because in calculation of reservation price it's not multiplied by the current price, therefore
        # it can't be a percentage. The result of the multiplication has to be an absolute price value because it's being subtracted from the current price
        vol = self.get_volatility()

        # order book liquidity - kappa and alpha have to represent absolute values because the second member of the optimal spread equation has to be an absolute price
        # and from the reservation price calculation we know that gamma's unit is not absolute price
        if all((self.gamma, self._kappa)) and self._alpha != 0 and self._kappa > 0 and vol != 0:
            if self._execution_state.time_left is not None and self._execution_state.closing_time is not None:
                # Avellaneda-Stoikov for a fixed timespan
                time_left_fraction = Decimal(str(self._execution_state.time_left / self._execution_state.closing_time))
            else:
                # Avellaneda-Stoikov for an infinite timespan
                # The equations in the paper for this contain a few mistakes
                # - the units don't align with the rest of the paper
                # - volatility cancels itself out completely
                # - the risk factor gets partially canceled
                # The proposed solution is to use the same equation as for the constrained timespan but with
                # a fixed time left
                time_left_fraction = 1

            # Here seems to be another mistake in the paper
            # It doesn't make sense to use mid_price_variance because its units would be absolute price units ^2, yet that side of the equation is subtracted
            # from the actual mid price of the asset in absolute price units
            # gamma / risk_factor gains a meaning of a fraction (percentage) of the volatility (standard deviation between ticks) to be subtraced from the
            # current mid price
            # This leads to normalization of the risk_factor and will guaranetee consistent behavior on all price ranges of the asset, and across assets

            self._reservation_price = price - (q * self.gamma * vol * time_left_fraction)

            self._optimal_spread = self.gamma * vol * time_left_fraction
            self._optimal_spread += 2 * Decimal(1 + self.gamma / self._kappa).ln() / self.gamma

            min_spread = price / 100 * Decimal(str(self._config_map.min_spread))

            max_limit_bid = price - min_spread / 2
            min_limit_ask = price + min_spread / 2

            self._optimal_ask = max(self._reservation_price + self._optimal_spread / 2, min_limit_ask)
            self._optimal_bid = min(self._reservation_price - self._optimal_spread / 2, max_limit_bid)

            # This is not what the algorithm will use as proposed bid and ask. This is just the raw output.
            # Optimal bid and optimal ask prices will be used
            if self._is_debug:
                self.logger().info(f"q={q:.4f} | "
                                   f"vol={vol:.10f}")
                self.logger().info(f"mid_price={price:.10f} | "
                                   f"reservation_price={self._reservation_price:.10f} | "
                                   f"optimal_spread={self._optimal_spread:.10f}")
                self.logger().info(f"optimal_bid={(price-(self._reservation_price - self._optimal_spread / 2)) / price * 100:.4f}% | "
                                   f"optimal_ask={((self._reservation_price + self._optimal_spread / 2) - price) / price * 100:.4f}%")

    def calculate_reservation_price_and_optimal_spread(self):
        return self.c_calculate_reservation_price_and_optimal_spread()

    cdef object c_calculate_target_inventory(self):
        cdef:
            ExchangeBase market = self._market_info.market
            str trading_pair = self._market_info.trading_pair
            object inventory_value
            object target_inventory_value

        price = self.get_price()
        # Total inventory value in quote asset prices 
        # in perpetual market, the total quote is the total account value
        inventory_value = self.get_quote_amount()
        # Target base asset value in quote asset prices
        target_inventory_value = inventory_value * self.inventory_target_base
        # Target base asset amount
        target_inventory_amount = target_inventory_value / price
        return market.c_quantize_order_amount(trading_pair, Decimal(str(target_inventory_amount)))

    def calculate_target_inventory(self) -> Decimal:
        return self.c_calculate_target_inventory()

    cdef c_calculate_inventory(self):
        cdef:
            object inventory_value_quote
            object inventory_value_base

        price = self.get_price()
        # Total inventory value in quote asset prices
        inventory_value_quote = self.get_quote_amount()
        # Total inventory value in base asset prices
        inventory_value_base = inventory_value_quote / price
        return inventory_value_base

    def calculate_inventory(self) -> Decimal:
        return self.c_calculate_inventory()

    cdef bint c_is_algorithm_ready(self):
        return self._avg_vol.is_sampling_buffer_full and self._trading_intensity.is_sampling_buffer_full

    cdef bint c_is_algorithm_changed(self):
        return self._trading_intensity.is_sampling_buffer_changed or self._avg_vol.is_sampling_buffer_changed

    def is_algorithm_ready(self) -> bool:
        return self.c_is_algorithm_ready()

    def is_algorithm_changed(self) -> bool:
        return self.c_is_algorithm_changed()

    def _get_level_spreads(self):
        level_step = ((self._optimal_spread / 2) / 100) * self.level_distances

        bid_level_spreads = [i * level_step for i in range(self.order_levels)]
        ask_level_spreads = [i * level_step for i in range(self.order_levels)]

        return bid_level_spreads, ask_level_spreads

    cdef _create_proposal_based_on_order_override(self):
        cdef:
            ExchangeBase market = self._market_info.market
            list buys = []
            list sells = []
        reference_price = self.get_price()
        if self.order_override is not None:
            for key, value in self.order_override.items():
                if str(value[0]) in ["buy", "sell"]:
                    list_to_be_appended = buys if str(value[0]) == "buy" else sells
                    size = Decimal(str(value[2]))
                    size = market.c_quantize_order_amount(self.trading_pair, size)
                    if str(value[0]) == "buy":
                        price = reference_price * (Decimal("1") - Decimal(str(value[1])) / Decimal("100"))
                    elif str(value[0]) == "sell":
                        price = reference_price * (Decimal("1") + Decimal(str(value[1])) / Decimal("100"))
                    price = market.c_quantize_order_price(self.trading_pair, price)
                    if size > 0 and price > 0:
                        list_to_be_appended.append(PriceSize(price, size))
        return buys, sells

    def create_proposal_based_on_order_override(self) -> Tuple[List[Proposal], List[Proposal]]:
        return self._create_proposal_based_on_order_override()

    cdef _create_proposal_based_on_order_levels(self):
        cdef:
            ExchangeBase market = self._market_info.market
            list buys = []
            list sells = []
        bid_level_spreads, ask_level_spreads = self._get_level_spreads()
        size = market.c_quantize_order_amount(self.trading_pair, self._config_map.order_amount)
        if size > 0:
            for level in range(self.order_levels):
                bid_price = market.c_quantize_order_price(self.trading_pair,
                                                          self._optimal_bid - Decimal(str(bid_level_spreads[level])))
                ask_price = market.c_quantize_order_price(self.trading_pair,
                                                          self._optimal_ask + Decimal(str(ask_level_spreads[level])))

                buys.append(PriceSize(bid_price, size))
                sells.append(PriceSize(ask_price, size))
        return buys, sells

    def create_proposal_based_on_order_levels(self):
        return self._create_proposal_based_on_order_levels()

    cdef _create_basic_proposal(self):
        cdef:
            ExchangeBase market = self._market_info.market
            list buys = []
            list sells = []
        price = market.c_quantize_order_price(self.trading_pair, Decimal(str(self._optimal_bid)))
        size = market.c_quantize_order_amount(self.trading_pair, self._config_map.order_amount)
        if size > 0:
            buys.append(PriceSize(price, size))

        price = market.c_quantize_order_price(self.trading_pair, Decimal(str(self._optimal_ask)))
        size = market.c_quantize_order_amount(self.trading_pair, self._config_map.order_amount)
        if size > 0:
            sells.append(PriceSize(price, size))
        return buys, sells

    def create_basic_proposal(self):
        return self._create_basic_proposal()

    cdef object c_create_base_proposal(self):
        cdef:
            ExchangeBase market = self._market_info.market
            list buys = []
            list sells = []

        if self.order_override is not None and len(self.order_override) > 0:
            # If order_override is set, it will override order_levels
            buys, sells = self._create_proposal_based_on_order_override()
        elif self.order_levels > 0:
            # Simple order levels
            buys, sells = self._create_proposal_based_on_order_levels()
        else:
            # No order levels nor order_overrides. Just 1 bid and 1 ask order
            buys, sells = self._create_basic_proposal()

        return Proposal(buys, sells)

    def create_base_proposal(self):
        return self.c_create_base_proposal()

    cdef c_apply_order_price_modifiers(self, object proposal):
        if self._config_map.order_optimization_enabled:
            self.c_apply_order_optimization(proposal)

        if self._config_map.add_transaction_costs:
            self.c_apply_add_transaction_costs(proposal)

    def apply_order_price_modifiers(self, proposal: Proposal):
        self.c_apply_order_price_modifiers(proposal)

    def get_position_action(self, side: TradeType) -> PositionAction:
        if self.direction == Direction.LONG:
            return PositionAction.CLOSE if side == TradeType.SELL else PositionAction.OPEN
        if self.direction == Direction.SHORT:
            return PositionAction.OPEN if side == TradeType.SELL else PositionAction.CLOSE
        return PositionAction.OPEN

    def get_position_close_limit(self, side: TradeType) -> Decimal:
        if side == TradeType.BUY:
            # get only the amount of the position that is in the direction of the trade to be closed
            # position close for buy is for short side, hence the negative sign
            return -self.get_base_amount()
        return self.get_base_amount()

    def create_order_candidates_for_budget_check(self, proposal: Proposal):
        order_candidates = []

        is_maker = True
        position_close = self.get_position_action(TradeType.BUY) == PositionAction.CLOSE
        if position_close:
            max_size = self.get_position_close_limit(TradeType.BUY)
            print(f"max_size: {max_size} position side: {TradeType.BUY}")
            for buy in proposal.buys:
                if buy.size > max_size:
                    buy.size = max_size
                    max_size = 0
                    continue
                max_size -= buy.size
        order_candidates.extend(
            [
                PerpetualOrderCandidate(
                    self.trading_pair,
                    is_maker,
                    OrderType.LIMIT,
                    TradeType.BUY,
                    buy.size,
                    buy.price,
                    leverage=Decimal(self.leverage),
                    position_close=position_close,
                )
                for buy in proposal.buys
            ]
        )
        position_close = self.get_position_action(TradeType.SELL) == PositionAction.CLOSE
        if position_close:
            max_size = self.get_position_close_limit(TradeType.SELL)
            print(f"max_size: {max_size} position side: {TradeType.SELL}")
            for sell in proposal.sells:
                if sell.size > max_size:
                    sell.size = max_size
                    max_size = 0
                    continue
                max_size -= sell.size
        order_candidates.extend(
            [
                PerpetualOrderCandidate(
                    self.trading_pair,
                    is_maker,
                    OrderType.LIMIT,
                    TradeType.SELL,
                    sell.size,
                    sell.price,
                    leverage=Decimal(self.leverage),
                    position_close=position_close,
                )
                for sell in proposal.sells
            ]
        )
        return order_candidates

    def apply_adjusted_order_candidates_to_proposal(self,
                                                    adjusted_candidates: List[PerpetualOrderCandidate],
                                                    proposal: Proposal):
        for order in chain(proposal.buys, proposal.sells):
            adjusted_candidate = adjusted_candidates.pop(0)
            if adjusted_candidate.amount == s_decimal_zero:
                self.logger().info(
                    f"Insufficient balance: {adjusted_candidate.order_side.name} order (price: {order.price},"
                    f" size: {order.size}) is omitted."
                )
                self.logger().warning(
                    "You are also at a possible risk of being liquidated if there happens to be an open loss.")
                order.size = s_decimal_zero
        proposal.buys = [o for o in proposal.buys if o.size > 0]
        proposal.sells = [o for o in proposal.sells if o.size > 0]

    def apply_budget_constraint(self, proposal: Proposal):
        checker = self._market_info.market.budget_checker

        order_candidates = self.create_order_candidates_for_budget_check(proposal)
        adjusted_candidates = checker.adjust_candidates(order_candidates, all_or_none=True)
        self.apply_adjusted_order_candidates_to_proposal(adjusted_candidates, proposal)

    # Compare the market price with the top bid and top ask price
    cdef c_apply_order_optimization(self, object proposal):
        cdef:
            ExchangeBase market = self._market_info.market
            object own_buy_size = s_decimal_zero
            object own_sell_size = s_decimal_zero
            object best_order_spread

        for order in self.active_orders:
            if order.is_buy:
                own_buy_size = order.quantity
            else:
                own_sell_size = order.quantity

        if len(proposal.buys) > 0:
            # Get the top bid price in the market using order_optimization_depth and your buy order volume
            top_bid_price = self._market_info.get_price_for_volume(
                False, own_buy_size).result_price
            price_quantum = market.c_get_order_price_quantum(
                self.trading_pair,
                top_bid_price
            )
            # Get the price above the top bid
            price_above_bid = (ceil(top_bid_price / price_quantum) + 1) * price_quantum

            # If the price_above_bid is lower than the price suggested by the top pricing proposal,
            # lower the price and from there apply the best_order_spread to each order in the next levels
            proposal.buys = sorted(proposal.buys, key = lambda p: p.price, reverse = True)
            for i, proposed in enumerate(proposal.buys):
                if proposal.buys[i].price > price_above_bid:
                    proposal.buys[i].price = market.c_quantize_order_price(self.trading_pair, price_above_bid)

        if len(proposal.sells) > 0:
            # Get the top ask price in the market using order_optimization_depth and your sell order volume
            top_ask_price = self._market_info.get_price_for_volume(
                True, own_sell_size).result_price
            price_quantum = market.c_get_order_price_quantum(
                self.trading_pair,
                top_ask_price
            )
            # Get the price below the top ask
            price_below_ask = (floor(top_ask_price / price_quantum) - 1) * price_quantum

            # If the price_below_ask is higher than the price suggested by the pricing proposal,
            # increase your price and from there apply the best_order_spread to each order in the next levels
            proposal.sells = sorted(proposal.sells, key = lambda p: p.price)
            for i, proposed in enumerate(proposal.sells):
                if proposal.sells[i].price < price_below_ask:
                    proposal.sells[i].price = market.c_quantize_order_price(self.trading_pair, price_below_ask)

    def apply_order_optimization(self, proposal: Proposal):
        return self.c_apply_order_optimization(proposal)

    cdef c_apply_order_amount_eta_transformation(self, object proposal):
        cdef:
            ExchangeBase market = self._market_info.market
            str trading_pair = self._market_info.trading_pair

        # Order amounts should be changed only if order_override is not active
        if (self.order_override is None) or (len(self.order_override) == 0):
            # eta parameter is described in the paper as the shape parameter for having exponentially decreasing order amount
            # for orders that go against inventory target (i.e. Want to buy when excess inventory or sell when deficit inventory)
            q = self.c_get_q_ratio()

            if len(proposal.buys) > 0:
                if q > 0:
                    for i, proposed in enumerate(proposal.buys):

                        proposal.buys[i].size = market.c_quantize_order_amount(trading_pair, proposal.buys[i].size * Decimal.exp(-self.eta * q))
                    proposal.buys = [o for o in proposal.buys if o.size > 0]

            if len(proposal.sells) > 0:
                if q < 0:
                    for i, proposed in enumerate(proposal.sells):
                        proposal.sells[i].size = market.c_quantize_order_amount(trading_pair, proposal.sells[i].size * Decimal.exp(self.eta * q))
                    proposal.sells = [o for o in proposal.sells if o.size > 0]

    def apply_order_amount_eta_transformation(self, proposal: Proposal):
        self.c_apply_order_amount_eta_transformation(proposal)

    cdef c_apply_add_transaction_costs(self, object proposal):
        cdef:
            ExchangeBase market = self._market_info.market
        for buy in proposal.buys:
            fee = build_perpetual_trade_fee(
                self.exchange_name,
                True,
                self.get_position_action(TradeType.BUY),
                self.base_asset, 
                self.quote_asset,
                self._limit_order_type,
                TradeType.BUY,
                buy.size,
                buy.price)

            price = buy.price * (Decimal(1) - fee.percent)
            buy.price = market.c_quantize_order_price(self.trading_pair, price)
        for sell in proposal.sells:
            fee = build_perpetual_trade_fee(
                self.exchange_name,
                True,
                self.get_position_action(TradeType.BUY),
                self.base_asset, 
                self.quote_asset,
                self._limit_order_type,
                TradeType.SELL, sell.size, sell.price)
            price = sell.price * (Decimal(1) + fee.percent)
            sell.price = market.c_quantize_order_price(self.trading_pair, price)

    def apply_add_transaction_costs(self, proposal: Proposal):
        self.c_apply_add_transaction_costs(proposal)

    cdef c_did_fill_order(self, object order_filled_event):
        cdef:
            str order_id = order_filled_event.order_id
            object market_info = self._sb_order_tracker.c_get_shadow_market_pair_from_order_id(order_id)
            tuple order_fill_record

        if market_info is not None:
            limit_order_record = self._sb_order_tracker.c_get_shadow_limit_order(order_id)
            order_fill_record = (limit_order_record, order_filled_event)

            if order_filled_event.trade_type is TradeType.BUY:
                if self._logging_options & self.OPTION_LOG_MAKER_ORDER_FILLED:
                    self.log_with_clock(
                        logging.INFO,
                        f"({market_info.trading_pair}) Maker buy order of "
                        f"{order_filled_event.amount} {market_info.base_asset} filled."
                    )
            else:
                if self._logging_options & self.OPTION_LOG_MAKER_ORDER_FILLED:
                    self.log_with_clock(
                        logging.INFO,
                        f"({market_info.trading_pair}) Maker sell order of "
                        f"{order_filled_event.amount} {market_info.base_asset} filled."
                    )

    cdef c_did_complete_buy_order(self, object order_completed_event):
        self.c_did_complete_order(order_completed_event)

    cdef c_did_complete_sell_order(self, object order_completed_event):
        self.c_did_complete_order(order_completed_event)

    cdef c_did_complete_order(self, object order_completed_event):
        cdef:
            str order_id = order_completed_event.order_id
            LimitOrder limit_order_record = self._sb_order_tracker.c_get_limit_order(self._market_info, order_id)

        if limit_order_record is None:
            return

        # Continue only if the order is not a hanging order
        if (not self._hanging_orders_tracker.is_order_id_in_hanging_orders(order_id)
                and not self.hanging_orders_tracker.is_order_id_in_completed_hanging_orders(order_id)):
            # delay order creation by filled_order_delay (in seconds)
            self._create_timestamp = self._current_timestamp + self.filled_order_delay
            self._cancel_timestamp = min(self._cancel_timestamp, self._create_timestamp)

            if limit_order_record.is_buy:
                self._filled_buys_balance += 1
                order_action_string = "buy"
            else:
                self._filled_sells_balance += 1
                order_action_string = "sell"

            self._last_own_trade_price = limit_order_record.price

            self.log_with_clock(
                logging.INFO,
                f"({self.trading_pair}) Maker {order_action_string} order {order_id} "
                f"({limit_order_record.quantity} {limit_order_record.base_currency} @ "
                f"{limit_order_record.price} {limit_order_record.quote_currency}) has been completely filled."
            )
            self.notify_hb_app_with_timestamp(
                f"Maker {order_action_string.upper()} order "
                f"{limit_order_record.quantity} {limit_order_record.base_currency} @ "
                f"{limit_order_record.price} {limit_order_record.quote_currency} is filled."
            )

    cdef bint c_is_within_tolerance(self, list current_prices, list proposal_prices):
        if len(current_prices) != len(proposal_prices):
            return False
        current_prices = sorted(current_prices)
        proposal_prices = sorted(proposal_prices)
        for current, proposal in zip(current_prices, proposal_prices):
            # if spread diff is more than the tolerance or order quantities are different, return false.
            if abs(proposal - current) / current > self.order_refresh_tolerance:
                return False
        return True

    def is_within_tolerance(self, current_prices: List[Decimal], proposal_prices: List[Decimal]) -> bool:
        return self.c_is_within_tolerance(current_prices, proposal_prices)

    cdef c_cancel_active_orders_on_max_age_limit(self):
        """
        Cancels active non hanging orders if they are older than max age limit
        """
        cdef:
            list active_orders = self.active_non_hanging_orders
        for order in active_orders:
            if order_age(order, self._current_timestamp) > self._config_map.max_order_age:
                self.c_cancel_order(self._market_info, order.client_order_id)

    cdef c_cancel_active_orders(self, object proposal):
        if self._cancel_timestamp > self._current_timestamp:
            return

        cdef:
            list active_buy_prices = []
            list active_sells = []
            bint to_defer_canceling = False

        if len(self.active_non_hanging_orders) == 0:
            return
        if proposal is not None:
            active_buy_prices = [Decimal(str(o.price)) for o in self.active_non_hanging_orders if o.is_buy]
            active_sell_prices = [Decimal(str(o.price)) for o in self.active_non_hanging_orders if not o.is_buy]
            proposal_buys = [buy.price for buy in proposal.buys]
            proposal_sells = [sell.price for sell in proposal.sells]

            if self.c_is_within_tolerance(active_buy_prices, proposal_buys) and \
                    self.c_is_within_tolerance(active_sell_prices, proposal_sells):
                to_defer_canceling = True

        if not to_defer_canceling:
            self._hanging_orders_tracker.update_strategy_orders_with_equivalent_orders()
            for order in self.active_non_hanging_orders:
                # If is about to be added to hanging_orders then don't cancel
                if not self._hanging_orders_tracker.is_potential_hanging_order(order):
                    self.c_cancel_order(self._market_info, order.client_order_id)
        else:
            self.c_set_timers()

    def cancel_active_orders(self, proposal: Proposal = None):
        return self.c_cancel_active_orders(proposal)

    cdef bint c_to_create_orders(self, object proposal):
        non_hanging_orders_non_cancelled = [o for o in self.active_non_hanging_orders if not
                                            self._hanging_orders_tracker.is_potential_hanging_order(o)]

        return (self._create_timestamp < self._current_timestamp
                and (not self._config_map.should_wait_order_cancel_confirmation or
                     len(self._sb_order_tracker.in_flight_cancels) == 0)
                and proposal is not None
                and len(non_hanging_orders_non_cancelled) == 0)

    def to_create_orders(self, proposal: Proposal) -> bool:
        return self.c_to_create_orders(proposal)

    cdef c_execute_orders_proposal(self, object proposal):
        cdef:
            double expiration_seconds = NaN
            str bid_order_id, ask_order_id
            bint orders_created = False
        # Number of pair of orders to track for hanging orders
        number_of_pairs = min((len(proposal.buys), len(proposal.sells))) if self._hanging_orders_enabled else 0
        if len(proposal.buys) > 0:
            if self._logging_options & self.OPTION_LOG_CREATE_ORDER:
                price_quote_str = [f"{buy.size.normalize()} {self.base_asset}, "
                                   f"{buy.price.normalize()} {self.quote_asset}"
                                   for buy in proposal.buys]
                self.logger().info(
                    f"({self.trading_pair}) Creating {len(proposal.buys)} bid orders "
                    f"at (Size, Price): {price_quote_str}"
                    f"position_action: {self.get_position_action(TradeType.BUY)}"
                )
            for idx, buy in enumerate(proposal.buys):
                bid_order_id = self.c_buy_with_specific_market(
                    self._market_info,
                    buy.size,
                    order_type=self._limit_order_type,
                    price=buy.price,
                    expiration_seconds=expiration_seconds,
                    position_action=self.get_position_action(TradeType.BUY),
                )
                orders_created = True
                if idx < number_of_pairs:
                    order = next((o for o in self.active_orders if o.client_order_id == bid_order_id))
                    if order:
                        self._hanging_orders_tracker.add_current_pairs_of_proposal_orders_executed_by_strategy(
                            CreatedPairOfOrders(order, None))
        if len(proposal.sells) > 0:
            if self._logging_options & self.OPTION_LOG_CREATE_ORDER:
                price_quote_str = [f"{sell.size.normalize()} {self.base_asset}, "
                                   f"{sell.price.normalize()} {self.quote_asset}"
                                   for sell in proposal.sells]
                self.logger().info(
                    f"({self.trading_pair}) Creating {len(proposal.sells)} ask "
                    f"orders at (Size, Price): {price_quote_str}"
                    f"position_action: {self.get_position_action(TradeType.SELL)}"
                )
            for idx, sell in enumerate(proposal.sells):
                ask_order_id = self.c_sell_with_specific_market(
                    self._market_info,
                    sell.size,
                    order_type=self._limit_order_type,
                    price=sell.price,
                    expiration_seconds=expiration_seconds,
                    position_action=self.get_position_action(TradeType.SELL),
                )
                orders_created = True
                if idx < number_of_pairs:
                    order = next((o for o in self.active_orders if o.client_order_id == ask_order_id))
                    if order:
                        self._hanging_orders_tracker.current_created_pairs_of_orders[idx].sell_order = order
        if orders_created:
            self.c_set_timers()

    def execute_orders_proposal(self, proposal: Proposal):
        self.c_execute_orders_proposal(proposal)

    cdef c_set_timers(self):
        cdef double next_cycle = self._current_timestamp + self.order_refresh_time
        if self._create_timestamp <= self._current_timestamp:
            self._create_timestamp = next_cycle
        if self._cancel_timestamp <= self._current_timestamp:
            self._cancel_timestamp = min(self._create_timestamp, next_cycle)

    def set_timers(self):
        self.c_set_timers()

    def notify_hb_app(self, msg: str):
        if self._hb_app_notification:
            super().notify_hb_app(msg)

    def dump_debug_variables(self):
        market = self._market_info.market
        mid_price = self.get_price()
        spread = Decimal(str(self.c_get_spread()))

        best_ask = mid_price + spread / 2
        new_ask = self._reservation_price + self._optimal_spread / 2
        best_bid = mid_price - spread / 2
        new_bid = self._reservation_price - self._optimal_spread / 2

        vol = self.get_volatility()
        mid_price_variance = vol ** 2

        if not os.path.exists(self._debug_csv_path):
            df_header = pd.DataFrame([('mid_price',
                                       'best_bid',
                                       'best_ask',
                                       'reservation_price',
                                       'optimal_spread',
                                       'optimal_bid',
                                       'optimal_ask',
                                       'optimal_bid_to_mid_%',
                                       'optimal_ask_to_mid_%',
                                       'current_inv',
                                       'target_inv',
                                       'time_left_fraction',
                                       'mid_price std_dev',
                                       'risk_factor',
                                       'gamma',
                                       'alpha',
                                       'kappa',
                                       'eta',
                                       'volatility',
                                       'mid_price_variance',
                                       'inventory_target_pct')])
            df_header.to_csv(self._debug_csv_path, mode='a', header=False, index=False)

        if self._execution_state.time_left is not None and self._execution_state.closing_time is not None:
            time_left_fraction = self._execution_state.time_left / self._execution_state.closing_time
        else:
            time_left_fraction = None

        df = pd.DataFrame([(mid_price,
                            best_bid,
                            best_ask,
                            self._reservation_price,
                            self._optimal_spread,
                            self._optimal_bid,
                            self._optimal_ask,
                            (mid_price - (self._reservation_price - self._optimal_spread / 2)) / mid_price,
                            ((self._reservation_price + self._optimal_spread / 2) - mid_price) / mid_price,
                            self.get_base_amount(),
                            self.c_calculate_target_inventory(),
                            time_left_fraction,
                            self._avg_vol.current_value,
                            self.gamma,
                            self._alpha,
                            self._kappa,
                            self.eta,
                            vol,
                            mid_price_variance,
                            self.inventory_target_base_pct)])
        df.to_csv(self._debug_csv_path, mode='a', header=False, index=False)
