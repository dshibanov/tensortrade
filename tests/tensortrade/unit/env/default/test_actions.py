from tensortrade.env.default.actions import TestActionScheme, StopLossPercent, TakeProfitPercent

def test_test_action_scheme():

    config = {'stop_loss_policy': StopLossPercent(percent=5),
              'take_profit_policy': TakeProfitPercent(percent=7)
             }

    actschm = TestActionScheme(config)






if __name__ == "__main__":
    test_test_action_scheme()

