using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class LMSAgent : Agent
{
    EnvironmentParameters m_ResetParams;
    BufferSensorComponent m_BufferSensor;

    int MINSIZE = 20;
    int MAXSIZE = 100;

    int iTaskSize;
    int[] vTaskMemory;
    int[] vTaskPower;
    bool[] vTaskAssinged;

    public override void Initialize()
    {
        m_ResetParams = Academy.Instance.EnvironmentParameters;
        m_BufferSensor = GetComponent<BufferSensorComponent>();
    }

    public override void OnEpisodeBegin()
    {
        InitEpisode();
    }

    void InitEpisode()
    {
        iTaskSize = Random.Range(MINSIZE, MAXSIZE);
        vTaskMemory = new int[iTaskSize];
        vTaskPower = new int[iTaskSize];
        vTaskAssinged = new bool[MAXSIZE];

        for (int i = 0; i < iTaskSize; ++i)
        {
            vTaskMemory[i] = Random.Range(10, 90); // MB
            vTaskPower[i] = Random.Range(15, 70); // Gigacycle/sec
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        for (int i = 0; i < iTaskSize; ++i)
        {
            float[] buf = new float[MAXSIZE + 2];
            buf[i] = 1.0f;
            buf[MAXSIZE] = vTaskMemory[i] * 1e-2f;
            buf[MAXSIZE + 1] = vTaskPower[i] * 1e-2f;
            m_BufferSensor.AppendObservation(buf);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        var preReward = GetReward();
        vTaskAssinged[actions.DiscreteActions[0]] = true;
        var afterReward = GetReward();

        //Debug.Log("----------------------------");
        //Debug.Log(afterReward);
        //for (int i = 0; i < iTaskSize; ++i)
        //{
        //    if(vTaskAssinged[i])
        //    {
        //        Debug.Log(vTaskMemory[i].ToString() + "/" + vTaskPower[i].ToString());
        //        break;
        //    }
        //}
        if (actions.DiscreteActions[0] >= iTaskSize)
        {
            SetReward(-1f);
        }
        else
        {
            //SetReward(100f * (afterReward - preReward) + 85f);
            SetReward(afterReward);
        }
    }

    float GetReward()
    {
        var Xtran_l = 1.2f;    // $/MB
        var Xtran_h = 3.0f;     // $/MB
        var Xcomp_l = 3.0f;     // $/Gigacycles
        var Xcomp_h = 10f;       // $/Gigacycles
        
        float result = 0;
        for (int i = 0; i < iTaskSize; ++i)
        {
            float Ttran = 0;
            float Ptran = 0;
            float Tcomp = 0;
            float Pcomp = 0;

            if (vTaskAssinged[i]) // LMS
            {
                var bandwidth = 2e+3f * (1f / 20f);    // MHz
                var power = 8e+2f * (1f / 20f);         // Gigacycles/sec
                var SNR = 1f;
                Ttran = vTaskMemory[i] / (bandwidth * Mathf.Log(1 + SNR, 2));
                Ptran = Xtran_l * bandwidth;
                Tcomp = vTaskPower[i] / power;
                Pcomp = Xcomp_l * power;
            }
            else // CNS
            {
                var bandwidth = 2e+3f * (1f / 100f);    // MHz
                var power = 5e+3f * (1f / 100f);         // Gigacycles/sec
                var SNR = 1f;
                Ttran = vTaskMemory[i] / (bandwidth * Mathf.Log(1 + SNR, 2));
                Ptran = Xtran_h * bandwidth;
                Tcomp = vTaskPower[i] / power;
                Pcomp = Xcomp_h * power;
            }

            float Tser = Ttran + Tcomp;
            float Pser = Ptran + Pcomp;
            result += (1f * Tser) + (1f * Pser);
        }
        return 2e+3f / result;
    }
}
