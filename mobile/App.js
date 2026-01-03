import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, TouchableOpacity, Alert, Platform, TextInput, ScrollView, SafeAreaView, Switch } from 'react-native';
import { useState, useEffect } from 'react';
// Stripe removed - see Purchases import below
// Default API URL - changeable in app
const DEFAULT_API_URL = process.env.EXPO_PUBLIC_API_URL || 'http://192.168.1.X:8899';

// Mock Data for Demo Mode
const MOCK_DEVICES = [
  { name: 'Yamaha AG06/AG03 (Demo)', id: 'mock_ag06' },
  { name: 'MacBook Pro Microphone (Demo)', id: 'mock_mic' }
];

function HardwareControl({ serverUrl, isDemoMode }) {
  const [devices, setDevices] = useState([]);
  const [loading, setLoading] = useState(false);
  const [monitorStatus, setMonitorStatus] = useState('');

  const scanDevices = async () => {
    setLoading(true);

    // DEMO MODE INTERCEPTION
    if (isDemoMode) {
      setTimeout(() => {
        setDevices(MOCK_DEVICES);
        Alert.alert('Demo Mode', 'Simulated AG06 detection successful.');
        setLoading(false);
      }, 1500); // Simulate network latency
      return;
    }

    try {
      // Use configured server URL
      const response = await fetch(`${serverUrl}/api/devices`);
      const data = await response.json();

      if (data.status === 'ok') {
        if (data.ag06_found) {
          Alert.alert('AG06 Found!', `Detected ${data.devices.length} Yamaha device(s).`);
          setDevices(data.devices);
        } else {
          Alert.alert('No AG06', `Server has no AG06 connected.\nFound ${data.all_devices.length} other devices.`);
          setDevices(data.all_devices);
        }
      } else {
        Alert.alert('Scan Failed', data.message);
      }
    } catch (e) {
      Alert.alert('Network Error', `Could not reach ${serverUrl}\n${e.message}`);
    }
    setLoading(false);
  };

  const startMonitor = async () => {
    setMonitorStatus('Requesting...');

    // DEMO MODE INTERCEPTION
    if (isDemoMode) {
      setTimeout(() => {
        setMonitorStatus('Monitoring (Demo: Active)');
        Alert.alert('Demo Mode', 'Simulated audio monitoring started.');
      }, 1000);
      return;
    }

    try {
      const response = await fetch(`${serverUrl}/api/monitor/start`, { method: 'POST' });
      const data = await response.json();
      if (data.status === 'started') {
        setMonitorStatus('Monitoring (Check Server Logs)');
        Alert.alert('Test Started', 'Audio monitoring started on server for 10s.');
      } else {
        setMonitorStatus('Failed');
      }
    } catch (e) {
      setMonitorStatus('Error');
      Alert.alert('Error', e.message);
    }
  };

  return (
    <View style={styles.hwContainer}>
      <Text style={styles.sectionTitle}>Audio Hardware</Text>

      <TouchableOpacity
        style={[styles.button, styles.scanButton]}
        onPress={scanDevices}
        disabled={loading}
      >
        <Text style={styles.buttonText}>
          {loading ? 'Scanning...' : isDemoMode ? 'Simulate Scan (Demo)' : 'Scan Devices on Server'}
        </Text>
      </TouchableOpacity>

      {devices.length > 0 && (
        <View style={styles.deviceList}>
          <Text style={styles.deviceListTitle}>Detected Devices:</Text>
          {devices.map((d, i) => (
            <Text key={i} style={styles.deviceItem}>• {d.name}</Text>
          ))}
        </View>
      )}

      <View style={styles.spacer} />

      <TouchableOpacity
        style={[styles.button, styles.monitorButton]}
        onPress={startMonitor}
      >
        <Text style={styles.buttonText}>
          {isDemoMode ? 'Simulate Monitor (Demo)' : 'Start Audio Monitor Test'}
        </Text>
      </TouchableOpacity>
      <Text style={styles.statusLabel}>{monitorStatus}</Text>
    </View>
  );
}

// IAP Integration via RevenueCat
import Purchases, { LOG_LEVEL } from 'react-native-purchases';

// RevenueCat Keys (Public SDK Keys are safe to expose in client)
const RC_KEY = process.env.EXPO_PUBLIC_RC_API_KEY || 'appl_placeholder_key';

function SubscriptionScreen() {
  const [loading, setLoading] = useState(false);
  const [proStatus, setProStatus] = useState('Free Plan');
  const [packages, setPackages] = useState([]);

  useEffect(() => {
    const initIAP = async () => {
      try {
        Purchases.setLogLevel(LOG_LEVEL.DEBUG);
        if (Platform.OS === 'ios') {
          Purchases.configure({ apiKey: RC_KEY });
        }

        // Get current offerings
        const offerings = await Purchases.getOfferings();
        if (offerings.current && offerings.current.availablePackages.length !== 0) {
          setPackages(offerings.current.availablePackages);
        }

        // Check active entitlement
        const customerInfo = await Purchases.getCustomerInfo();
        if (customerInfo.entitlements.active['Pro Access']) {
          setProStatus('Pro Unlocked 🌟');
        }
      } catch (e) {
        console.log('IAP Init Error:', e);
      }
    };
    initIAP();
  }, []);

  const purchase = async (pkg) => {
    setLoading(true);
    try {
      const { customerInfo } = await Purchases.purchasePackage(pkg);
      if (customerInfo.entitlements.active['Pro Access']) {
        setProStatus('Pro Unlocked 🌟');
        Alert.alert('Success', 'Welcome to Pro!');
      }
    } catch (e) {
      if (!e.userCancelled) {
        Alert.alert('Purchase Error', e.message);
      }
    }
    setLoading(false);
  };

  const restore = async () => {
    setLoading(true);
    try {
      const customerInfo = await Purchases.restorePurchases();
      if (customerInfo.entitlements.active['Pro Access']) {
        setProStatus('Pro Unlocked 🌟');
        Alert.alert('Restored', 'Your purchases have been restored.');
      } else {
        Alert.alert('Notice', 'No active subscriptions found to restore.');
      }
    } catch (e) {
      Alert.alert('Error', e.message);
    }
    setLoading(false);
  };

  return (
    <View style={styles.paymentContainer}>
      <Text style={styles.sectionTitle}>Unlock Pro Features</Text>
      <Text style={styles.statusLabel}>Status: {proStatus}</Text>

      {/* Dynamic Package List or Default Button */}
      {packages.length > 0 ? (
        packages.map((pkg) => (
          <TouchableOpacity
            key={pkg.identifier}
            style={[styles.button, styles.subscribeButton, { marginBottom: 10 }]}
            onPress={() => purchase(pkg)}
            disabled={loading || proStatus.includes('Unlocked')}
          >
            <Text style={styles.buttonText}>
              {loading ? 'Processing...' : `Subscribe ${pkg.product.priceString}`}
            </Text>
          </TouchableOpacity>
        ))
      ) : (
        <TouchableOpacity
          style={[styles.button, styles.subscribeButton]}
          onPress={() => Alert.alert('Config Missing', 'Connecting to App Store...')}
          disabled={loading}
        >
          <Text style={styles.buttonText}>
            Subscribe ($9.99/mo)
          </Text>
        </TouchableOpacity>
      )}

      {/* Mandatory Restore Button */}
      <TouchableOpacity onPress={restore} style={{ marginTop: 15 }}>
        <Text style={{ color: '#aaa', textDecorationLine: 'underline' }}>
          Restore Purchases
        </Text>
      </TouchableOpacity>
    </View>
  );
}

export default function App() {
  const [status, setStatus] = useState('Disconnected');
  const [serverUrl, setServerUrl] = useState('http://192.168.1.X:8899'); // Default placeholder
  const [isDemoMode, setIsDemoMode] = useState(false); // Demo Mode State

  useEffect(() => {
    // Try to auto-detect or use env logic if available
  }, []);

  const testConnection = async () => {
    if (isDemoMode) {
      setStatus('Connected (Demo Mode)');
      Alert.alert('Demo Mode', 'Simulating connection to backend v1.0.0');
      return;
    }

    try {
      setStatus('Connecting...');
      const response = await fetch(`${serverUrl}/api/health`);
      if (response.ok) {
        const data = await response.json();
        setStatus(`Connected ✅ (${data.version})`);
        Alert.alert('Success', `Connected to backend v${data.version}`);
      } else {
        setStatus('Error: ' + response.status);
      }
    } catch (error) {
      setStatus('Failed: ' + error.message);
      console.error(error);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>AiOke Studio</Text>
        <Text style={styles.subtitle}>AG06 Mobile Controller</Text>

        {/* Network Configuration */}
        <View style={styles.configContainer}>
          <View style={styles.row}>
            <Text style={styles.label}>Server URL:</Text>
            <View style={styles.demoContainer}>
              <Text style={styles.demoLabel}>Demo Mode</Text>
              <Switch
                value={isDemoMode}
                onValueChange={setIsDemoMode}
                trackColor={{ false: "#767577", true: "#81b0ff" }}
                thumbColor={isDemoMode ? "#f5dd4b" : "#f4f3f4"}
              />
            </View>
          </View>

          <TextInput
            style={[styles.input, isDemoMode && styles.disabledInput]}
            value={serverUrl}
            onChangeText={setServerUrl}
            placeholder="http://192.168.1.X:8899"
            placeholderTextColor="#666"
            autoCapitalize="none"
            autoCorrect={false}
            editable={!isDemoMode}
          />
          {isDemoMode && <Text style={styles.hintText}>Demo Mode enabled for App Store Review</Text>}
        </View>

        <View style={styles.statusContainer}>
          <Text style={styles.statusLabel}>Backend Status:</Text>
          <Text style={styles.statusValue}>{status}</Text>
        </View>

        <TouchableOpacity style={styles.button} onPress={testConnection}>
          <Text style={styles.buttonText}>{isDemoMode ? 'Test Connection (Simulated)' : 'Test Connection'}</Text>
        </TouchableOpacity>

        <View style={styles.divider} />

        <HardwareControl serverUrl={serverUrl} isDemoMode={isDemoMode} />

        <View style={styles.divider} />

        <SubscriptionScreen />

        <StatusBar style="light" />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#1a1a1a',
  },
  container: {
    flexGrow: 1,
    backgroundColor: '#1a1a1a',
    alignItems: 'center',
    paddingVertical: 50,
  },
  title: {
    fontSize: 42,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 18,
    color: '#aaa',
    marginBottom: 30,
  },
  configContainer: {
    width: '80%',
    marginBottom: 20,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  demoContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  demoLabel: {
    color: '#ccc',
    marginRight: 10,
    fontSize: 12,
  },
  label: {
    color: '#ccc',
    fontSize: 14,
  },
  input: {
    backgroundColor: '#333',
    color: '#fff',
    padding: 15,
    borderRadius: 10,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#444',
  },
  disabledInput: {
    opacity: 0.5,
    backgroundColor: '#222',
  },
  hintText: {
    color: '#4caf50',
    fontSize: 12,
    marginTop: 5,
    fontStyle: 'italic',
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  statusLabel: {
    color: '#ccc',
    marginRight: 10,
    fontSize: 16,
  },
  statusValue: {
    color: '#4caf50',
    fontSize: 16,
    fontWeight: 'bold',
  },
  button: {
    backgroundColor: '#2196f3',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
    minWidth: 200,
    alignItems: 'center',
  },
  scanButton: {
    backgroundColor: '#ff9800', // Orange for hardware
    marginBottom: 10,
  },
  monitorButton: {
    backgroundColor: '#4caf50', // Green for action
    marginTop: 20,
  },
  subscribeButton: {
    backgroundColor: '#e91e63', // Pink for subscribe
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  spacer: {
    height: 20,
  },
  sectionTitle: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  hwContainer: {
    width: '100%',
    alignItems: 'center',
  },
  paymentContainer: {
    width: '100%',
    alignItems: 'center',
    marginTop: 10,
  },
  deviceList: {
    backgroundColor: '#333',
    padding: 10,
    borderRadius: 10,
    width: '80%',
    marginTop: 10,
  },
  deviceListTitle: {
    color: '#aaa',
    fontSize: 14,
    marginBottom: 5,
  },
  deviceItem: {
    color: '#fff',
    fontSize: 14,
    fontFamily: Platform.OS === 'ios' ? 'Courier' : 'monospace',
  },
  divider: {
    height: 1,
    backgroundColor: '#444',
    width: '80%',
    marginVertical: 30,
  },
});

